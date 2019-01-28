/**
 * Single Producer Single Consumer Lock Free Queue
 * fmannan@gmail.com
 */
#include <iostream>
#include <thread>
#include <memory>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <array>


template<typename T, size_t BufferSize>
class SPSC_LockFreeQueue {
public:
  SPSC_LockFreeQueue() {
    writer_pos = 0;
    reader_pos = 0;
  }
  SPSC_LockFreeQueue(const SPSC_LockFreeQueue&) = delete;
  SPSC_LockFreeQueue& operator=(const SPSC_LockFreeQueue&) = delete;

  bool push(T& data) {
    _data[writer_pos] = std::move(data);
    writer_pos = getPositionAfter(writer_pos);
  }

  bool pop_latest(T& data) {
    if(empty())
      return false;
    size_t pos = getPositionBefore(writer_pos);
    data = std::move(_data[pos]);
    reader_pos = pos;
    return true;
  }
  
  bool pop(T& data) {
    if(empty())
      return false;
    data = std::move(_data[reader_pos.load()]);
    reader_pos = getPositionAfter(reader_pos);
    return true;
  }
  
  bool empty() const {
    return reader_pos == writer_pos;
  }

  static constexpr size_t getPositionAfter(size_t pos) noexcept {
    return ((pos + 1) == BufferSize) ? 0 : pos + 1;
  }

  static constexpr size_t getPositionBefore(size_t pos) noexcept {
    return ((pos - 1) == -1) ? (BufferSize - 1) : pos - 1;
  }

private:
  std::array<T, BufferSize> _data;
  std::atomic<size_t> reader_pos = {0};
  std::atomic<size_t> writer_pos = {0};
};
