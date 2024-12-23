#include "allocator/tlsf.hpp"

#include <semaphore.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "metadata.hpp"

TlsfAllocator::TlsfAllocator(TopicHeader* topic_header, bool read_only,
                             const std::string& sock_file_dir)
        : Allocator(topic_header, read_only, sock_file_dir) {
    // NOTE: `tlsf_header` is already initialized in Node's constructor
    this->tlsf_header = getTlsfHeader(topic_header);
    this->blocks =
        (TlsfBlockMetadata*) ((uintptr_t) tlsf_header + sizeof(TlsfHeader));

    bool pool_inited =
        std::atomic_ref<bool>(this->tlsf_header->inited).exchange(true);
    if (!pool_inited) {
        this->blocks[0].header =
            tlsf_header->aligned_pool_size | kBlockFreeFlag | kBlockLastFlag;
        this->blocks[0].phys_prev_idx = -1;
        this->insertBlock(0);
    }
}

size_t TlsfAllocator::malloc(size_t size) {
    if (size == 0) return -1;

    size = this->alignSize(size);
    if (size > this->tlsf_header->aligned_pool_size) return -1;

    int fidx, sidx;
    this->mapping(size, &fidx, &sidx);

    sem_wait(&this->tlsf_header->lock);
    size_t block_idx = findSuitableBlock(size, &fidx, &sidx);
    if (block_idx == -1) {
        sem_post(&this->tlsf_header->lock);
        return -1;
    }
    this->removeBlock(block_idx, fidx, sidx);

    if (this->blocks[block_idx].getSize() > size) {
        size_t rest_block_idx = this->splitBlock(block_idx, size);
        this->insertBlock(rest_block_idx);
    }

    sem_post(&this->tlsf_header->lock);
    return block_idx * kBlockMinSize;
}

void TlsfAllocator::free(size_t offset) {
    if (offset == -1) return;
    size_t block_idx = offset / kBlockMinSize;
    sem_wait(&this->tlsf_header->lock);
    if (this->blocks[block_idx].header & kBlockFreeFlag) {
        sem_post(&this->tlsf_header->lock);
        return;
    }
    block_idx = this->mergeBlock(block_idx);
    this->blocks[block_idx].header |= kBlockFreeFlag;
    this->insertBlock(block_idx);
    sem_post(&this->tlsf_header->lock);
}

void TlsfAllocator::mapping(size_t size, int* fidx, int* sidx) const {
    // index of leftmost 1-bit
    *fidx = kWidthMinusOne - __builtin_clzll(size);
    // `kSndLvlIdx` bits from the right of the leftmost 1-bit
    *sidx = (size ^ (1 << *fidx)) >> (*fidx - kSndLvlIdx);
}

// align `size` to the multiple of `kBlockMinSize` (2^`kSndLvlIdx`)
size_t TlsfAllocator::alignSize(size_t size) const {
    return (((size - 1) >> kSndLvlIdx) + 1) << kSndLvlIdx;
}

size_t TlsfAllocator::findSuitableBlock(size_t size, int* fidx, int* sidx) {
    // non-empty lists indexed by `*fidx` and second-level indices greater than
    // `*sidx`
    uint32_t non_empty_lists =
        this->tlsf_header->second_lvl[*fidx] & (~0U << (*sidx + 1));

    if (!non_empty_lists) {
        // let's look for larger blocks in first level
        uint32_t non_empty_groups =
            this->tlsf_header->first_lvl & (~0U << (*fidx + 1));
        // no suitable block
        if (!non_empty_groups) return -1;
        *fidx = __builtin_ffs(non_empty_groups) - 1;
        non_empty_lists = this->tlsf_header->second_lvl[*fidx];
    }

    *sidx = __builtin_ffs(non_empty_lists) - 1;
    return this->tlsf_header->free_lists[*fidx][*sidx] - 1;
}

size_t TlsfAllocator::splitBlock(size_t block_idx, size_t size) {
    size_t rest_block_idx = block_idx + size / kBlockMinSize;
    TlsfBlockMetadata* block = this->blocks + block_idx;
    TlsfBlockMetadata* rest_block = this->blocks + rest_block_idx;

    rest_block->header = (block->getSize() - size) | kBlockFreeFlag;
    rest_block->phys_prev_idx = block_idx;
    block->header = size | (block->header & kBlockFlagBits);
    if (block->header & kBlockLastFlag) {
        block->header ^= kBlockLastFlag;
        rest_block->header |= kBlockLastFlag;
    } else {
        (rest_block + rest_block->getSize() / kBlockMinSize)->phys_prev_idx =
            rest_block_idx;
    }

    return rest_block_idx;
}

size_t TlsfAllocator::mergeBlock(size_t block_idx) {
    TlsfBlockMetadata* block = this->blocks + block_idx;

    size_t prev_idx = block->phys_prev_idx;
    size_t next_idx = block_idx + block->getSize() / kBlockMinSize;
    TlsfBlockMetadata* prev = this->blocks + prev_idx;
    TlsfBlockMetadata* next = this->blocks + next_idx;

    size_t new_block_idx = block_idx;
    bool block_is_last = block->header & kBlockLastFlag;

    if (!block_is_last && (next->header & kBlockFreeFlag)) {
        // IMPORTANT: remove block should be done before the size transfer
        this->removeBlock(next_idx);
        block->header += next->getSize();
        block_is_last = next->header & kBlockLastFlag;
    }
    if (prev_idx != -1 && (prev->header & kBlockFreeFlag)) {
        // IMPORTANT: remove block should be done before the size transfer
        this->removeBlock(prev_idx);
        prev->header += block->getSize();
        new_block_idx = prev_idx;
    }

    TlsfBlockMetadata* new_block = this->blocks + new_block_idx;
    if (block_is_last) {
        new_block->header |= kBlockLastFlag;
    } else {
        (new_block + new_block->getSize() / kBlockMinSize)->phys_prev_idx =
            new_block_idx;
    }

    return new_block_idx;
}

// insert the block to the head of the list
void TlsfAllocator::insertBlock(size_t block_idx) {
    TlsfBlockMetadata* block = this->blocks + block_idx;
    int fidx, sidx;
    this->mapping(block->getSize(), &fidx, &sidx);

    this->tlsf_header->first_lvl |= 1 << fidx;
    this->tlsf_header->second_lvl[fidx] |= 1 << sidx;

    size_t head_idx = this->tlsf_header->free_lists[fidx][sidx] - 1;
    if (head_idx != -1) this->blocks[head_idx].prev_free_idx = block_idx;
    block->prev_free_idx = -1;
    block->next_free_idx = head_idx;
    this->tlsf_header->free_lists[fidx][sidx] = block_idx + 1;
}

// remove the block from the list
void TlsfAllocator::removeBlock(size_t block_idx) {
    int fidx, sidx;
    this->mapping(this->blocks[block_idx].getSize(), &fidx, &sidx);
    this->removeBlock(block_idx, fidx, sidx);
}

void TlsfAllocator::removeBlock(size_t block_idx, int fidx, int sidx) {
    size_t prev_idx = this->blocks[block_idx].prev_free_idx;
    size_t next_idx = this->blocks[block_idx].next_free_idx;

    if (prev_idx != -1) this->blocks[prev_idx].next_free_idx = next_idx;
    if (next_idx != -1) this->blocks[next_idx].prev_free_idx = prev_idx;

    if (this->tlsf_header->free_lists[fidx][sidx] - 1 == block_idx) {
        this->tlsf_header->free_lists[fidx][sidx] = next_idx + 1;

        if (next_idx == -1) {
            this->tlsf_header->second_lvl[fidx] &= ~(1U << sidx);
            if (!this->tlsf_header->second_lvl[fidx]) {
                this->tlsf_header->first_lvl &= ~(1U << fidx);
            }
        }
    }

    this->blocks[block_idx].header &= ~kBlockFreeFlag;
}
