#include "alloc_algo/tlsf.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

// NOTE: `tlsf_header` is initialized in Node's constructor
Tlsf::Tlsf(Header* tlsf_header) : tlsf_header(tlsf_header) {
    this->blocks = (BlockMetadata*) ((uintptr_t) tlsf_header + sizeof(Header));

    bool pool_inited = std::atomic_ref<bool>(tlsf_header->inited).exchange(true);
    if (!pool_inited) {
        this->blocks[0].header = tlsf_header->aligned_pool_size | kBlockFreeFlag;
        this->insertBlock(0);
    }
}

size_t Tlsf::malloc(size_t size) {
    if (size == 0 || size > this->tlsf_header->aligned_pool_size) return -1;

    size = this->alignSize(size);
    int fidx, sidx;
    this->mapping(size, &fidx, &sidx);

    // prevent the found block from being removed in mergeBlock()
    this->tlsf_header->lock.lock();
    size_t block_idx = findSuitableBlock(size, &fidx, &sidx);
    if (block_idx == -1) {
        this->tlsf_header->lock.unlock();
        return -1;
    }
    this->removeBlock(block_idx, fidx, sidx);
    this->tlsf_header->lock.unlock();

    if (this->blocks[block_idx].getSize() > size) {
        size_t rest_block_idx = this->splitBlock(block_idx, size);
        this->insertBlock(rest_block_idx);
    }

    return block_idx * kBlockMinSize;
}

void Tlsf::free(size_t offset) {
    if (offset == -1) return;
    size_t block_idx = offset / kBlockMinSize;
    if (this->blocks[block_idx].header & kBlockFreeFlag) return;
    block_idx = this->mergeBlock(block_idx);
    this->blocks[block_idx].header |= kBlockFreeFlag;
    this->insertBlock(block_idx);
}

size_t Tlsf::BlockMetadata::getSize() const {
    return this->header & ~kBlockFlagBits;
}

void Tlsf::mapping(size_t size, int* fidx, int* sidx) const {
    // index of leftmost 1-bit 
    *fidx = kWidthMinusOne - __builtin_clzll(size);
    // `kSndLvlIdx` bits from the right of the leftmost 1-bit
    *sidx = (size ^ (1 << *fidx)) >> (*fidx - kSndLvlIdx);
}

// align `size` to the multiple of `kBlockMinSize` (2^`kSndLvlIdx`)
size_t Tlsf::alignSize(size_t size) const {
    return (((size - 1) >> kSndLvlIdx) + 1) << kSndLvlIdx;
}

size_t Tlsf::findSuitableBlock(size_t size, int* fidx, int* sidx) {
    // non-empty lists indexed by `*fidx` and second-level indices not less than `*sidx`
    uint32_t non_empty_lists = this->tlsf_header->second_lvl[*fidx] & (~0U << *sidx);

    if (!non_empty_lists) {
        // let's look for larger blocks in first level
        uint32_t non_empty_groups = this->tlsf_header->first_lvl & (~0U << (*fidx + 1));
        // no suitable block
        if (!non_empty_groups) return -1;
        *fidx = __builtin_ffs(non_empty_groups) - 1;
        non_empty_lists = this->tlsf_header->second_lvl[*fidx];
    }

    *sidx = __builtin_ffs(non_empty_lists) - 1;
    return this->tlsf_header->free_lists[*fidx][*sidx];
}

size_t Tlsf::splitBlock(size_t block_idx, size_t size) {
    size_t rest_block_idx = block_idx + size / kBlockMinSize;
    BlockMetadata* block = this->blocks + block_idx;
    BlockMetadata* rest_block = this->blocks + rest_block_idx;

    rest_block->header = (block->getSize() - size) | kBlockFreeFlag;
    block->header = size | (block->header & kBlockFlagBits);
    return rest_block_idx;
}

size_t Tlsf::mergeBlock(size_t block_idx) {
    BlockMetadata* block = this->blocks + block_idx;
    BlockMetadata* prev = block - 1;
    BlockMetadata* next = block + 1;
    size_t new_block_idx = block_idx;

    // prevent next block or previous block chosen by findSuitableBlock() before removeBlock() is called
    this->tlsf_header->lock.lock();
    if (block_idx < this->tlsf_header->block_count - 1 && (next->header & kBlockFreeFlag)) {
        block->header += next->getSize();
        this->removeBlock(block_idx + 1);
    }
    if (block_idx > 0 && (prev->header & kBlockFreeFlag)) {
        prev->header += block->getSize();
        this->removeBlock(block_idx);
        new_block_idx = block_idx - 1;
    }
    this->tlsf_header->lock.unlock();

    return new_block_idx;
}

// insert the block to the head of the list
void Tlsf::insertBlock(size_t block_idx) {
    BlockMetadata* block = this->blocks + block_idx;
    int fidx, sidx;
    this->mapping(block->getSize(), &fidx, &sidx);

    this->tlsf_header->lock.lock();
    this->tlsf_header->first_lvl |= 1 << fidx;
    this->tlsf_header->second_lvl[fidx] |= 1 << sidx;

    size_t head_idx = this->tlsf_header->free_lists[fidx][sidx];
    if (head_idx != -1) this->blocks[head_idx].prev_free_idx = block_idx;
    block->prev_free_idx = -1;
    block->next_free_idx = head_idx;
    this->tlsf_header->free_lists[fidx][sidx] = block_idx;
    this->tlsf_header->lock.unlock();
}

// remove the block from the list
void Tlsf::removeBlock(size_t block_idx) {
    int fidx, sidx;
    this->mapping(this->blocks[block_idx].getSize(), &fidx, &sidx);
    this->removeBlock(block_idx, fidx, sidx);
}

void Tlsf::removeBlock(size_t block_idx, int fidx, int sidx) {
    // NOTE: Why the following 5 lines should be included in the critical section is because
    // when neighoring blocks are removed at the same time, the list may break into 2 parts.
    // E.g., 2 & 3 are removed: [1]<->[2]<->[3]<->[4] => [1]<->[3] [2]<->[4]
    size_t prev_idx = this->blocks[block_idx].prev_free_idx;
    size_t next_idx = this->blocks[block_idx].next_free_idx;

    if (prev_idx != -1) this->blocks[prev_idx].next_free_idx = next_idx;
    if (next_idx != -1) this->blocks[next_idx].prev_free_idx = prev_idx;

    if (this->tlsf_header->free_lists[fidx][sidx] == block_idx) {
        this->tlsf_header->free_lists[fidx][sidx] = next_idx;

        if (next_idx == -1) {
            this->tlsf_header->second_lvl[fidx] &= ~(1U << sidx);
            if (!this->tlsf_header->second_lvl[fidx]) {
                this->tlsf_header->first_lvl &= ~(1U << fidx);
            }
        }
    }

    this->blocks[block_idx].header &= ~kBlockFreeFlag;
}