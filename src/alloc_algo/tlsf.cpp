#include "alloc_algo/tlsf.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

Tlsf::Tlsf(void* pool_base, size_t pool_size) : pool_base(pool_base) {
    this->first_lvl = 0;
    memset(this->second_lvl, 0, sizeof(this->second_lvl));
    memset(this->free_lists, 0, sizeof(this->free_lists));

    // NOTE: If `pool_size` is not a multiple of `kBlockMinSize`, the remaining space,
    //       whose size is less than `kBlockMinSize` will be wasted.
    this->block_count = pool_size / kBlockMinSize;
    this->blocks = new Tlsf::Block[this->block_count];
    this->pool_size = this->block_count * kBlockMinSize;

    this->blocks[0].header = pool_size | kBlockFreeFlag;
    this->insertBlock(&(this->blocks[0]));
}

Tlsf::~Tlsf() {
    delete[] this->blocks;
}

void* Tlsf::malloc(size_t size) {
    if (size == 0 || size > this->pool_size) return nullptr;
    size = this->alignSize(size);

    int fidx, sidx;
    this->mapping(size, &fidx, &sidx);
    Tlsf::Block* block = findSuitableBlock(size, &fidx, &sidx);
    if (!block) return nullptr;
    this->removeBlock(block, fidx, sidx);

    if (block->getSize() > size) {
        Tlsf::Block* rest_block = this->splitBlock(block, size);
        this->insertBlock(rest_block);
    }

    size_t offset = block - this->blocks;
    return (void*) ((uint8_t*) this->pool_base + offset * kBlockMinSize);
}

void Tlsf::free(void* addr) {
    if (!addr) return;
    Tlsf::Block* block = this->getBlockFromPayload(addr);
    block->header |= kBlockFreeFlag;
    block = this->mergeBlock(block);
    this->insertBlock(block);
}

size_t Tlsf::Block::getSize() const {
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

Tlsf::Block* Tlsf::getBlockFromPayload(void* addr) {
    size_t offset = ((uint8_t*) addr - (uint8_t*) this->pool_base) / kBlockMinSize;
    return this->blocks + offset;
}

Tlsf::Block* Tlsf::findSuitableBlock(size_t size, int* fidx, int* sidx) {
    // non-empty lists indexed by `*fidx` and second-level indices not less than `*sidx`
    uint32_t non_empty_lists = this->second_lvl[*fidx] & (~0U << *sidx);

    if (!non_empty_lists) {
        // let's look for larger blocks in first level
        uint32_t non_empty_groups = this->first_lvl & (~0U << (*fidx + 1));
        // no suitable block
        if (!non_empty_groups) return nullptr;
        *fidx = __builtin_ffs(non_empty_groups) - 1;
        non_empty_lists = this->second_lvl[*fidx];
    }

    *sidx = __builtin_ffs(non_empty_lists) - 1;
    return this->free_lists[*fidx][*sidx];
}

Tlsf::Block* Tlsf::splitBlock(Tlsf::Block* block, size_t size) {
    Tlsf::Block* rest_block = block + size / kBlockMinSize;
    rest_block->header = (block->getSize() - size) | kBlockFreeFlag;
    block->header = size | (block->header & kBlockFlagBits);
    return rest_block;
}

Tlsf::Block* Tlsf::mergeBlock(Tlsf::Block* block) {
    size_t offset = block - this->blocks;
    Tlsf::Block* next = block + 1;
    Tlsf::Block* prev = block - 1;
    Tlsf::Block* new_block = block;

    if (offset < this->block_count - 1 && (next->header & kBlockFreeFlag)) {
        block->header += next->getSize();
        this->removeBlock(next);
    }
    if (offset > 0 && (prev->header & kBlockFreeFlag)) {
        prev->header += block->getSize();
        this->removeBlock(block);
        new_block = prev;
    }

    return new_block;
}

// insert the block to the head of the list
void Tlsf::insertBlock(Tlsf::Block* block) {
    int fidx, sidx;
    this->mapping(block->getSize(), &fidx, &sidx);

    this->first_lvl |= 1 << fidx;
    this->second_lvl[fidx] |= 1 << sidx;

    Tlsf::Block* head = this->free_lists[fidx][sidx];
    if (head) head->prev_free = block;
    block->prev_free = nullptr;
    block->next_free = head;
    this->free_lists[fidx][sidx] = block;
}

// remove the block from the list
void Tlsf::removeBlock(Tlsf::Block* block) {
    int fidx, sidx;
    this->mapping(block->getSize(), &fidx, &sidx);
    this->removeBlock(block, fidx, sidx);
}

void Tlsf::removeBlock(Tlsf::Block* block, int fidx, int sidx) {
    Tlsf::Block* prev = block->prev_free;
    Tlsf::Block* next = block->next_free;

    if (prev) prev->next_free = next;
    if (next) next->prev_free = prev;

    if (this->free_lists[fidx][sidx] == block) {
        this->free_lists[fidx][sidx] = next;

        if (!next) {
            this->second_lvl[fidx] &= ~(1U << sidx);
            if (!this->second_lvl[fidx]) this->first_lvl &= ~(1U << fidx);
        }
    }

    block->header &= ~kBlockFreeFlag;
}