// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NOVA_ROS__TASK_QUEUE_HPP_
#define NOVA_ROS__TASK_QUEUE_HPP_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <utility>

namespace nova::ros
{
/**
 * @brief TaskQueue class.
 *
 * This class represents a task queue that can be used to manage tasks.
 */
class TaskQueue
{
public:
  /**
   * @brief Constructor for TaskQueue.
   *
   * @param queue_size The maximum size of the task queue.
   */
  explicit TaskQueue(size_t queue_size = 10) : queue_size_(queue_size) {}

  /**
   * @brief Add a task to the queue.
   *
   * @param task The task function to be added.
   */
  void add_task(std::function<void()> task)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.push(std::move(task));
    if (tasks_.size() > queue_size_) {
      // if the queue is too long, drop the oldest task
      tasks_.pop();
    }
    condition_.notify_one();
  }

  /**
   * @brief Run the task queue.
   *
   * This function runs the task queue until it is stopped.
   */
  void run()
  {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return request_stop_ || tasks_.empty(); });
        if (request_stop_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  /**
   * @brief Stop the task queue.
   *
   * This function stops the task queue.
   */
  void stop()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    request_stop_ = true;
    condition_.notify_one();
  }

private:
  std::queue<std::function<void()>> tasks_;  //!< The task queue.
  std::mutex mutex_;                         //!< The mutex for thread safety.
  std::condition_variable condition_;        //!< The condition variable for thread synchronization.
  bool request_stop_{false};                 //!< Flag indicating whether the task queue is stopped.
  size_t queue_size_;                        //!< The maximum size of the task queue.
};
}  // namespace nova::ros
#endif  // NOVA_ROS__TASK_QUEUE_HPP_
