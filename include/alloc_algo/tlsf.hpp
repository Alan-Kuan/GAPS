#ifndef TLSF_HPP
#define TLSF_HPP

#include <semaphore.h>

#include <cstddef>
#include <cstdint>

class Tlsf {
public:
    static constexpr int kWidthMinusOne = (sizeof(size_t) << 3) - 1;

    static const size_t kBlockFlagBits = 0b11;
    static const size_t kBlockFreeFlag = 0b01;
    static const size_t kBlockLastFlag = 0b10;

    static const int kFstLvlCnt = 32;
    static const int kSndLvlIdx = 4;
    static constexpr int kSndLvlCnt = 1 << kSndLvlIdx;
    static const size_t kBlockMinSize = 16;

    class BlockMetadata {
    public:
        friend class Tlsf;

        size_t getSize() const;

    private:
        // block size and free bit (LSB is used as free bit)
        size_t header;
        size_t phys_prev_idx;
        size_t prev_free_idx;
        size_t next_free_idx;
    };

    struct Header {
        sem_t lock;
        // whether the pool has been initialized (should be atomic referenced)
        bool inited;
        // aligned to a multiple of the minimum block size
        size_t aligned_pool_size;
        // number of blocks in this pool
        size_t block_count;
        // indicate if any block exists in the group of lists indexed by `fidx`
        uint32_t first_lvl;
        // indicate if any block exists in the list indexed by `fidx` and `sidx`
        uint32_t second_lvl[kFstLvlCnt];
        // free lists of blocks (saved in the form of "block_idx + 1")
        size_t free_lists[kFstLvlCnt][kSndLvlCnt];
    };

    Tlsf() = delete;
    Tlsf(Header* tlsf_header);

    size_t malloc(size_t size);
    void free(size_t offset);

private:
    void mapping(size_t size, int* fidx, int* sidx) const;
    size_t alignSize(size_t size) const;

    size_t findSuitableBlock(size_t size, int* fidx, int* sidx);

    size_t splitBlock(size_t idx, size_t size);
    size_t mergeBlock(size_t idx);

    void insertBlock(size_t idx);
    void removeBlock(size_t idx);
    void removeBlock(size_t idx, int fidx, int sidx);

    Header* tlsf_header = nullptr;
    BlockMetadata* blocks = nullptr;
};

#endif  // TLSF_HPP