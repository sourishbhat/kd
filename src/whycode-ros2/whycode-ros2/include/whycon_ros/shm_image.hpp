/*
shm_image.hpp -- single-producer / single-consumer shared-memory ring for raw
camera frames, bypassing DDS entirely.

WHY THIS EXISTS
---------------
/image_raw is 11MB per frame at 1920x1920 rgb8. Carrying that over DDS costs,
per frame: a serialize out of the publisher's buffer, a copy into the Fast-DDS
shared-memory segment, and a deserialize into the subscriber's message -- and,
because the publisher is RELIABLE with KeepLast(1), a subscriber that falls
behind fills the segment and applies BACKPRESSURE to publish(), which stalls
the render thread that called it. The segment also has to be sized by hand in
config/fastdds_profile.xml, and a segment too small for the payload silently
drops the topic back to UDPv4.

This ring removes all of that. The producer packs pixels STRAIGHT INTO the ring
slot (the pack pass has to happen anyway, so it costs nothing extra), and the
consumer memcpys once out of the slot into its own working buffer (which it
also has to do anyway -- whycode draws its overlays in place). Net: two passes
over the frame instead of five, no backpressure path, no segment to size.

WHAT IT IS NOT
--------------
Not a general pub/sub. Exactly one writer and (practically) one reader, same
machine, same ABI. There is no discovery, no QoS, no history: a reader only
ever sees the MOST RECENT completed frame, and frames produced while it was
busy are dropped on the floor. That is the correct semantics for a
vision-in-the-loop controller -- a stale frame is worse than no frame -- but it
is the wrong tool for anything that needs every sample.

HOW IT STAYS CORRECT WITHOUT LOCKS
----------------------------------
Each slot carries a seqlock counter. The writer makes it odd before touching
the slot's pixels and even again after, both with release ordering; the reader
loads it (acquire) before and after its copy and retries if the two differ or
if either was odd. So a reader can never hand back a half-written frame -- at
worst it retries. No mutex is held across the copy, so a reader that is
descheduled mid-copy cannot stall the writer for even an instant. This is the
one property a mutex could not give us: the render thread must never wait on
the vision node.

kSlots=3 is what makes retries essentially unreachable rather than merely
correct: the writer has to lap the reader by three full frames before it can
land on the slot being read. At a ~16ms production period against a ~2ms
copy, that does not happen.

KEEPING THE TWO COPIES IN SYNC
------------------------------
This file is DUPLICATED, byte for byte, in two separate git repositories:

    swift_pico/config/shm_image.hpp                     (producer)
    whycode-ros2/include/whycon_ros/shm_image.hpp       (consumer)

They are separate repos, and whycode has no business depending on swift_pico,
so there is no shared package to put this in. EDIT ONE, COPY TO THE OTHER.
kVersion below exists precisely because that instruction will eventually be
missed: the reader refuses to attach to a segment whose magic or version does
not match, so a divergence is a startup error rather than silently misread
pixels. BUMP kVersion whenever the layout of Header or SlotHeader changes.
*/

#ifndef SWIFT_PICO_SHM_IMAGE_HPP
#define SWIFT_PICO_SHM_IMAGE_HPP

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace shm_image
{

// 'SPI1' -- checked on attach so a reader cannot mistake some other process's
// segment (or a half-initialised one) for a frame ring.
constexpr std::uint32_t kMagic = 0x53504931u;

// Bump on ANY layout change to Header or SlotHeader. See the sync note above.
constexpr std::uint32_t kVersion = 1u;

// See the kSlots rationale in the file header.
constexpr std::uint32_t kSlots = 3u;

// Pixel planes start here, and each slot's plane is padded up to a multiple of
// this too, so no two slots ever share a cache line and the writer's stores to
// one slot cannot false-share with the reader's loads from another.
constexpr std::size_t kAlign = 64u;

// Shared across processes, so the atomics have to be genuinely lock-free --
// a libstdc++ fallback to a lock table would put the lock in each process's
// OWN address space and silently protect nothing.
static_assert(
  std::atomic<std::uint64_t>::is_always_lock_free,
  "shm_image needs lock-free 64-bit atomics for cross-process seqlocks");

struct SlotHeader
{
  // Even = stable and readable, odd = writer is mid-frame. See the seqlock
  // note in the file header.
  std::atomic<std::uint64_t> seq;
  // MuJoCo sim time of the pixels in this slot -- NOT wall time, and not the
  // time the frame was written. The producer's readback is pipelined, so the
  // pixels are a frame older than the sim state at the moment of the write;
  // the producer is responsible for passing the correct one.
  double sim_time;
  // Monotonic index of this frame across the whole run. The reader keeps the
  // last one it returned so it can tell a genuinely new frame from a re-read
  // of the same slot.
  std::uint64_t frame_index;
  char _pad[kAlign - ((sizeof(std::atomic<std::uint64_t>) + sizeof(double) +
                       sizeof(std::uint64_t)) % kAlign)];
};

struct Header
{
  // Written LAST by the producer (release), read FIRST by the consumer
  // (acquire), so the rest of this struct is guaranteed visible and final
  // before any reader accepts the segment.
  std::atomic<std::uint32_t> magic;
  std::uint32_t version;

  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t channels;
  std::uint32_t step;        // bytes per row == width * channels
  std::uint32_t slots;
  std::uint32_t _pad0;
  std::uint64_t frame_bytes; // step * height

  // Count of frames the producer has COMPLETED. The most recent one lives in
  // slot ((latest - 1) % slots). Zero means nothing has been published yet.
  std::atomic<std::uint64_t> latest;

  // Cleared by the producer on a clean shutdown. A reader uses it only to log
  // a tidier message; correctness never depends on it, because a producer that
  // is SIGKILLed never gets to clear it.
  std::atomic<std::uint32_t> writer_alive;
  std::uint32_t _pad1;

  SlotHeader slot[kSlots];
};

inline std::size_t align_up(std::size_t n)
{
  return (n + kAlign - 1u) & ~(kAlign - 1u);
}

inline std::size_t data_offset()
{
  return align_up(sizeof(Header));
}

inline std::size_t segment_bytes(std::size_t frame_bytes)
{
  return data_offset() + kSlots * align_up(frame_bytes);
}

// POSIX shm names must be "/name" with no further slashes.
inline std::string normalize_name(const std::string & name)
{
  if (!name.empty() && name[0] == '/') {return name;}
  return "/" + name;
}

// ── Producer ────────────────────────────────────────────────────────────────
//
// Usage per frame:
//
//     unsigned char * dst = writer.begin_frame();   // slot marked in-progress
//     ... fill exactly frame_bytes() at dst ...
//     writer.commit_frame(sim_time);                // slot published
//
// begin_frame must be paired with exactly one commit_frame. Dropping a frame
// after begin_frame (e.g. the render failed) is fine -- call abort_frame so
// the slot's seq returns to even and the slot is reused next time.
class Writer
{
public:
  Writer() = default;
  ~Writer() {close();}

  Writer(const Writer &) = delete;
  Writer & operator=(const Writer &) = delete;

  bool open(const std::string & name, int width, int height, int channels)
  {
    close();

    name_ = normalize_name(name);
    const std::size_t step = static_cast<std::size_t>(width) * channels;
    frame_bytes_ = step * static_cast<std::size_t>(height);
    const std::size_t total = segment_bytes(frame_bytes_);

    // Unlink any leftover segment from a previous run before creating ours.
    // Without this, a crashed producer leaves a segment sized for the OLD
    // resolution behind; ftruncate on an existing mapping would not resize it
    // for a reader that is still attached, and we would write past the end.
    shm_unlink(name_.c_str());

    fd_ = shm_open(name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd_ < 0) {
      std::fprintf(
        stderr, "shm_image: shm_open('%s') failed: %s\n", name_.c_str(), std::strerror(errno));
      return false;
    }
    if (ftruncate(fd_, static_cast<off_t>(total)) != 0) {
      std::fprintf(stderr, "shm_image: ftruncate(%zu) failed: %s\n", total, std::strerror(errno));
      close();
      return false;
    }

    void * p = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) {
      std::fprintf(stderr, "shm_image: mmap failed: %s\n", std::strerror(errno));
      close();
      return false;
    }
    base_ = static_cast<unsigned char *>(p);
    mapped_bytes_ = total;
    hdr_ = new (base_) Header{};

    hdr_->version = kVersion;
    hdr_->width = static_cast<std::uint32_t>(width);
    hdr_->height = static_cast<std::uint32_t>(height);
    hdr_->channels = static_cast<std::uint32_t>(channels);
    hdr_->step = static_cast<std::uint32_t>(step);
    hdr_->slots = kSlots;
    hdr_->frame_bytes = frame_bytes_;
    hdr_->latest.store(0, std::memory_order_relaxed);
    hdr_->writer_alive.store(1, std::memory_order_relaxed);
    for (std::uint32_t i = 0; i < kSlots; ++i) {
      hdr_->slot[i].seq.store(0, std::memory_order_relaxed);
      hdr_->slot[i].sim_time = 0.0;
      hdr_->slot[i].frame_index = 0;
    }

    // Everything above must be visible before any reader accepts the segment.
    hdr_->magic.store(kMagic, std::memory_order_release);
    return true;
  }

  // Marks the next slot in-progress and returns its pixel plane. Never blocks.
  unsigned char * begin_frame()
  {
    if (!hdr_) {return nullptr;}
    writing_slot_ = static_cast<std::uint32_t>(write_count_ % kSlots);
    SlotHeader & s = hdr_->slot[writing_slot_];
    // Odd: readers that load this will retry rather than trust the pixels.
    s.seq.fetch_add(1, std::memory_order_acq_rel);
    return slot_data(writing_slot_);
  }

  // Publishes the slot opened by begin_frame.
  void commit_frame(double sim_time)
  {
    if (!hdr_ || writing_slot_ == kInvalidSlot) {return;}
    SlotHeader & s = hdr_->slot[writing_slot_];
    s.sim_time = sim_time;
    s.frame_index = write_count_ + 1;
    // Back to even. Release pairs with the reader's acquire load, so the
    // pixels written above are visible to anyone who sees this value.
    s.seq.fetch_add(1, std::memory_order_release);
    ++write_count_;
    hdr_->latest.store(write_count_, std::memory_order_release);
    writing_slot_ = kInvalidSlot;
  }

  // Abandons the slot opened by begin_frame without publishing it.
  void abort_frame()
  {
    if (!hdr_ || writing_slot_ == kInvalidSlot) {return;}
    hdr_->slot[writing_slot_].seq.fetch_add(1, std::memory_order_release);
    writing_slot_ = kInvalidSlot;
  }

  void close()
  {
    if (hdr_) {
      hdr_->writer_alive.store(0, std::memory_order_release);
    }
    if (base_) {
      munmap(base_, mapped_bytes_);
      base_ = nullptr;
      hdr_ = nullptr;
      mapped_bytes_ = 0;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    if (!name_.empty()) {
      shm_unlink(name_.c_str());
      name_.clear();
    }
  }

  bool is_open() const {return hdr_ != nullptr;}
  std::size_t frame_bytes() const {return frame_bytes_;}
  const std::string & name() const {return name_;}

private:
  static constexpr std::uint32_t kInvalidSlot = 0xFFFFFFFFu;

  unsigned char * slot_data(std::uint32_t i)
  {
    return base_ + data_offset() + static_cast<std::size_t>(i) * align_up(frame_bytes_);
  }

  std::string name_;
  int fd_ = -1;
  unsigned char * base_ = nullptr;
  Header * hdr_ = nullptr;
  std::size_t mapped_bytes_ = 0;
  std::size_t frame_bytes_ = 0;
  std::uint64_t write_count_ = 0;
  std::uint32_t writing_slot_ = kInvalidSlot;
};

// ── Consumer ────────────────────────────────────────────────────────────────
//
// try_attach() is safe to call every poll: it is a no-op once attached, and
// cheap (one shm_open attempt) while it is not, so a consumer started before
// the producer simply keeps polling until the segment shows up.
class Reader
{
public:
  Reader() = default;
  ~Reader() {detach();}

  Reader(const Reader &) = delete;
  Reader & operator=(const Reader &) = delete;

  void configure(const std::string & name) {name_ = normalize_name(name);}

  bool attached() const {return hdr_ != nullptr;}

  int width() const {return hdr_ ? static_cast<int>(hdr_->width) : 0;}
  int height() const {return hdr_ ? static_cast<int>(hdr_->height) : 0;}
  int channels() const {return hdr_ ? static_cast<int>(hdr_->channels) : 0;}
  int step() const {return hdr_ ? static_cast<int>(hdr_->step) : 0;}
  std::size_t frame_bytes() const {return frame_bytes_; }

  // Returns true the first time it succeeds, and on every call thereafter.
  bool try_attach()
  {
    if (hdr_) {return true;}
    if (name_.empty()) {return false;}

    int fd = shm_open(name_.c_str(), O_RDONLY, 0);
    if (fd < 0) {return false;}  // producer not up yet -- not an error

    // Map the header alone first: the segment's real size is not known until
    // its dimensions have been read out, and a producer that is still inside
    // ftruncate may briefly present a zero-length file.
    struct stat st{};
    if (fstat(fd, &st) != 0 || static_cast<std::size_t>(st.st_size) < sizeof(Header)) {
      ::close(fd);
      return false;
    }

    void * p = mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
      ::close(fd);
      return false;
    }

    auto * h = static_cast<Header *>(p);
    // Acquire: pairs with the producer's release store of magic, so every
    // other field below is guaranteed to be the final value, not a torn one.
    if (h->magic.load(std::memory_order_acquire) != kMagic) {
      munmap(p, static_cast<std::size_t>(st.st_size));
      ::close(fd);
      return false;  // still initialising, or not our segment -- keep polling
    }
    if (h->version != kVersion || h->slots != kSlots) {
      std::fprintf(
        stderr,
        "shm_image: '%s' is version %u/%u slots, this build expects %u/%u -- "
        "the two copies of shm_image.hpp have diverged; re-sync them.\n",
        name_.c_str(), h->version, h->slots, kVersion, kSlots);
      munmap(p, static_cast<std::size_t>(st.st_size));
      ::close(fd);
      return false;
    }

    const std::size_t need = segment_bytes(h->frame_bytes);
    if (static_cast<std::size_t>(st.st_size) < need) {
      munmap(p, static_cast<std::size_t>(st.st_size));
      ::close(fd);
      return false;  // producer still growing the file
    }

    fd_ = fd;
    base_ = static_cast<unsigned char *>(p);
    mapped_bytes_ = static_cast<std::size_t>(st.st_size);
    hdr_ = h;
    frame_bytes_ = h->frame_bytes;
    // Start from whatever is current rather than from 0: a consumer that
    // attaches mid-run wants the newest frame, not a replay of slot 0.
    last_index_ = 0;
    last_progress_ = std::chrono::steady_clock::now();
    return true;
  }

  // Copies the most recent completed frame into dst.
  //
  // Returns false -- with nothing written -- when there is no frame NEWER than
  // the last one returned. That is the common case and is not an error: it is
  // what lets a caller poll far faster than the producer without doing work.
  bool read_latest(unsigned char * dst, std::size_t dst_bytes, double * sim_time_out)
  {
    if (!hdr_ || !dst) {return false;}
    if (dst_bytes < frame_bytes_) {return false;}

    const std::uint64_t latest = hdr_->latest.load(std::memory_order_acquire);
    if (latest == 0 || latest == last_index_) {
      maybe_reattach();
      return false;
    }

    const std::uint32_t slot = static_cast<std::uint32_t>((latest - 1) % kSlots);
    const SlotHeader & s = hdr_->slot[slot];

    // Seqlock: bracket the copy with two acquire loads of the slot counter and
    // reject anything odd (writer mid-frame) or changed (writer lapped us).
    // Retries are practically unreachable at kSlots=3 -- see the file header --
    // so a few attempts is plenty; failing means the producer is running far
    // ahead, and the right answer is to give up on THIS frame and let the
    // caller come back for a newer one.
    for (int attempt = 0; attempt < 4; ++attempt) {
      const std::uint64_t seq_before = s.seq.load(std::memory_order_acquire);
      if (seq_before & 1u) {continue;}

      const double sim_time = s.sim_time;
      const std::uint64_t index = s.frame_index;
      std::memcpy(dst, slot_data(slot), frame_bytes_);

      std::atomic_thread_fence(std::memory_order_acquire);
      if (s.seq.load(std::memory_order_acquire) != seq_before) {continue;}

      last_index_ = index;
      last_progress_ = std::chrono::steady_clock::now();
      if (sim_time_out) {*sim_time_out = sim_time;}
      return true;
    }
    return false;
  }

  void detach()
  {
    if (base_) {
      munmap(base_, mapped_bytes_);
      base_ = nullptr;
      hdr_ = nullptr;
      mapped_bytes_ = 0;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    frame_bytes_ = 0;
    last_index_ = 0;
  }

private:
  const unsigned char * slot_data(std::uint32_t i) const
  {
    return base_ + data_offset() + static_cast<std::size_t>(i) * align_up(frame_bytes_);
  }

  // A producer restart unlinks the old segment and creates a new one, but our
  // mapping still points at the old (now anonymous) pages, which are frozen
  // forever: `latest` stops advancing and we would wait for a frame that can
  // never arrive. Silence for longer than kStaleTimeout is the only observable
  // symptom, so treat it as "the producer went away" and re-attach by name.
  // A merely slow producer re-attaches to the same segment, which is harmless.
  void maybe_reattach()
  {
    constexpr std::chrono::seconds kStaleTimeout{2};
    if (std::chrono::steady_clock::now() - last_progress_ < kStaleTimeout) {return;}
    detach();
    last_progress_ = std::chrono::steady_clock::now();
    try_attach();
  }

  std::string name_;
  int fd_ = -1;
  unsigned char * base_ = nullptr;
  Header * hdr_ = nullptr;
  std::size_t mapped_bytes_ = 0;
  std::size_t frame_bytes_ = 0;
  std::uint64_t last_index_ = 0;
  std::chrono::steady_clock::time_point last_progress_{};
};

}  // namespace shm_image

#endif  // SWIFT_PICO_SHM_IMAGE_HPP
