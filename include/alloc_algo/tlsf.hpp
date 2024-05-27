#ifndef TLSF_HPP
#define TLSF_HPP

#include <cstddef>
#include <cstdint>

class Tlsf {
public:
    Tlsf(void* pool_base, size_t pool_size);
    ~Tlsf();

    void* malloc(size_t size);
    void free(void* addr);

private:
    class Block {
    public:
        friend class Tlsf;

        size_t getSize() const;

    private:
        // block size and free bit (LSB is used as free bit)
        size_t header;
        Block* prev_free;
        Block* next_free;
    };

    static constexpr int kWidthMinusOne = (sizeof(size_t) << 3) - 1;

    static const size_t kBlockFlagBits = 0b1;
    static const size_t kBlockFreeFlag = 0b1;

    static const int kFstLvlCnt = 32;
    static const int kSndLvlIdx = 4;
    static constexpr int kSndLvlCnt = 1 << kSndLvlIdx;
    static constexpr size_t kBlockMinSize = 1 << kSndLvlIdx;

    void mapping(size_t size, int* fidx, int* sidx) const;
    size_t alignSize(size_t size) const;

    Block* getBlockFromPayload(void* addr);
    Block* findSuitableBlock(size_t size, int* fidx, int* sidx);

    Block* splitBlock(Block* block, size_t size);
    Block* mergeBlock(Block* block);

    void insertBlock(Block* block);
    void removeBlock(Block* block);
    void removeBlock(Block* block, int fidx, int sidx);

    void* pool_base;
    size_t pool_size;
    // indicate if any block exists in the group of lists indexed by `fidx`
    uint32_t first_lvl;
    // indicate if any block exists in the list indexed by `fidx` and `sidx`
    uint32_t second_lvl[kFstLvlCnt];
    // free lists of blocks
    Block* free_lists[kFstLvlCnt][kSndLvlCnt];
    // where block is actually placed
    Block* blocks;
    size_t block_count;
};

#endif  // TLSF_HPP