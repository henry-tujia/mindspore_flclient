/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ps/core/communicator/tcp_client.h"

#include <arpa/inet.h>
#include <event2/buffer.h>
#include <event2/buffer_compat.h>
#include <event2/bufferevent.h>
#include <event2/event.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>

namespace mindspore {
namespace ps {
namespace core {
event_base *TcpClient::event_base_ = nullptr;
std::mutex TcpClient::event_base_mutex_;
bool TcpClient::is_started_ = false;

TcpClient::TcpClient(const std::string &address, std::uint16_t port, Configuration *const config)
    : event_timeout_(nullptr),
      buffer_event_(nullptr),
      server_address_(std::move(address)),
      server_port_(port),
      is_stop_(true),
      is_connected_(false),
      config_(config) {
  message_handler_.SetCallback(
    [this](const std::shared_ptr<MessageMeta> &meta, const Protos &protos, const void *data, size_t size) {
      if (message_callback_) {
        message_callback_(meta, protos, data, size);
      }
    });
}

TcpClient::~TcpClient() {
  if (buffer_event_) {
    bufferevent_free(buffer_event_);
    buffer_event_ = nullptr;
  }
  if (event_timeout_) {
    event_free(event_timeout_);
    event_timeout_ = nullptr;
  }
}

std::string TcpClient::GetServerAddress() const { return server_address_; }

void TcpClient::set_disconnected_callback(const OnDisconnected &disconnected) { disconnected_callback_ = disconnected; }

void TcpClient::set_connected_callback(const OnConnected &connected) { connected_callback_ = connected; }

bool TcpClient::WaitConnected(const uint32_t &connected_timeout) {
  std::unique_lock<std::mutex> lock(connection_mutex_);
  bool res = connection_cond_.wait_for(lock, std::chrono::seconds(connected_timeout),
                                       [this] { return this->is_connected_.load(); });
  return res;
}

void TcpClient::Init() {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  if (buffer_event_) {
    bufferevent_free(buffer_event_);
    buffer_event_ = nullptr;
  }
  if (!CommUtil::CheckIp(server_address_)) {
    MS_LOG(EXCEPTION) << "The tcp client ip:" << server_address_ << " is illegal!";
  }

  int result = evthread_use_pthreads();
  if (result != 0) {
    MS_LOG(EXCEPTION) << "Use event pthread failed!";
  }
  if (event_base_ == nullptr) {
    event_base_ = event_base_new();
    MS_EXCEPTION_IF_NULL(event_base_);
  }

  sockaddr_in sin{};
  if (memset_s(&sin, sizeof(sin), 0, sizeof(sin)) != EOK) {
    MS_LOG(EXCEPTION) << "Initialize sockaddr_in failed!";
  }
  sin.sin_family = AF_INET;
  sin.sin_addr.s_addr = inet_addr(server_address_.c_str());
  sin.sin_port = htons(server_port_);

  if (!PSContext::instance()->enable_ssl()) {
    MS_LOG(INFO) << "SSL is disable.";
    buffer_event_ = bufferevent_socket_new(event_base_, -1, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
  } else {
    if (!EstablishSSL()) {
      MS_LOG(WARNING) << "Establish SSL failed.";
      return;
    }
  }

  MS_EXCEPTION_IF_NULL(buffer_event_);

  bufferevent_setcb(buffer_event_, ReadCallback, nullptr, EventCallback, this);
  if (bufferevent_enable(buffer_event_, EV_READ | EV_WRITE) == -1) {
    MS_LOG(EXCEPTION) << "Buffer event enable read and write failed!";
  }

  int result_code = bufferevent_socket_connect(buffer_event_, reinterpret_cast<struct sockaddr *>(&sin), sizeof(sin));
  if (result_code < 0) {
    MS_LOG(EXCEPTION) << "Connect server ip:" << server_address_ << " and port: " << server_port_ << " is failed!";
  }
}

void TcpClient::StartWithDelay(int seconds) {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  if (buffer_event_) {
    return;
  }

  event_base_ = event_base_new();
  MS_EXCEPTION_IF_NULL(event_base_);

  timeval timeout_value{};
  timeout_value.tv_sec = seconds;
  timeout_value.tv_usec = 0;

  event_timeout_ = evtimer_new(event_base_, TimeoutCallback, this);
  MS_EXCEPTION_IF_NULL(event_timeout_);
  if (evtimer_add(event_timeout_, &timeout_value) == -1) {
    MS_LOG(EXCEPTION) << "Event timeout failed!";
  }
}

void TcpClient::Stop() {
  MS_EXCEPTION_IF_NULL(event_base_);
  std::lock_guard<std::mutex> lock(connection_mutex_);
  MS_LOG(INFO) << "Stop tcp client!";
  int ret = event_base_loopbreak(event_base_);
  if (ret != 0) {
    MS_LOG(ERROR) << "Event base loop break failed!";
  }
}

void TcpClient::SetTcpNoDelay(const evutil_socket_t &fd) {
  const int one = 1;
  int ret = setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(int));
  if (ret < 0) {
    MS_LOG(EXCEPTION) << "Set socket no delay failed!";
  }
}

void TcpClient::TimeoutCallback(evutil_socket_t, std::int16_t, void *const arg) {
  try {
    TimeoutCallbackInner(arg);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Catch exception: " << e.what();
  }
}

void TcpClient::TimeoutCallbackInner(void *const arg) {
  MS_EXCEPTION_IF_NULL(arg);
  auto tcp_client = reinterpret_cast<TcpClient *>(arg);
  tcp_client->Init();
}

void TcpClient::ReadCallback(struct bufferevent *bev, void *const ctx) {
  try {
    ReadCallbackInner(bev, ctx);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Catch exception: " << e.what();
  }
}

void TcpClient::ReadCallbackInner(struct bufferevent *bev, void *const ctx) {
  MS_EXCEPTION_IF_NULL(bev);
  MS_EXCEPTION_IF_NULL(ctx);
  auto tcp_client = reinterpret_cast<TcpClient *>(ctx);

  char read_buffer[kMessageChunkLength];
  int read = 0;

  while ((read = bufferevent_read(bev, &read_buffer, SizeToInt(sizeof(read_buffer)))) > 0) {
    tcp_client->OnReadHandler(read_buffer, IntToSize(read));
  }
}

void TcpClient::OnReadHandler(const void *buf, size_t num) {
  MS_EXCEPTION_IF_NULL(buf);
  if (read_callback_) {
    read_callback_(buf, num);
  }
  message_handler_.ReceiveMessage(buf, num);
}

void TcpClient::TimerCallback(evutil_socket_t, int16_t, void *arg) {
  MS_EXCEPTION_IF_NULL(arg);
  auto tcp_client = reinterpret_cast<TcpClient *>(arg);
  if (tcp_client->on_timer_callback_) {
    tcp_client->on_timer_callback_();
  }
}

void TcpClient::NotifyConnected() {
  MS_LOG(INFO) << "Client connected to the server!";
  is_connected_ = true;
  connection_cond_.notify_all();
}

bool TcpClient::EstablishSSL() {
  MS_LOG(INFO) << "Enable ssl support.";

  SSL *ssl = SSL_new(SSLClient::GetInstance().GetSSLCtx());
  MS_ERROR_IF_NULL_W_RET_VAL(ssl, false);
  MS_ERROR_IF_NULL_W_RET_VAL(event_base_, false);

  buffer_event_ = bufferevent_openssl_socket_new(event_base_, -1, ssl, BUFFEREVENT_SSL_CONNECTING,
                                                 BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
  return true;
}

void TcpClient::EventCallback(struct bufferevent *bev, std::int16_t events, void *const ptr) {
  try {
    EventCallbackInner(bev, events, ptr);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Catch exception: " << e.what();
  }
}

void TcpClient::EventCallbackInner(struct bufferevent *bev, std::int16_t events, void *const ptr) {
  MS_EXCEPTION_IF_NULL(bev);
  MS_EXCEPTION_IF_NULL(ptr);
  auto tcp_client = reinterpret_cast<TcpClient *>(ptr);
  if (events & BEV_EVENT_CONNECTED) {
    // Connected
    if (tcp_client->connected_callback_) {
      tcp_client->connected_callback_();
    }
    tcp_client->NotifyConnected();
    evutil_socket_t fd = bufferevent_getfd(bev);
    SetTcpNoDelay(fd);
    MS_LOG(INFO) << "Client connected!";
  } else if (events & BEV_EVENT_ERROR) {
    MS_LOG(WARNING) << "The client will retry to connect to the server!";
    if (tcp_client->disconnected_callback_) {
      tcp_client->disconnected_callback_();
    }
  } else if (events & BEV_EVENT_EOF) {
    MS_LOG(WARNING) << "Client connected end of file";
    if (tcp_client->disconnected_callback_) {
      tcp_client->disconnected_callback_();
    }
  }
}

void TcpClient::Start() {
  event_base_mutex_.lock();
  if (is_started_) {
    event_base_mutex_.unlock();
    return;
  }
  is_started_ = true;
  event_base_mutex_.unlock();
  MS_EXCEPTION_IF_NULL(event_base_);
  int ret = event_base_dispatch(event_base_);
  // is_started_ should be false when finish dispatch
  is_started_ = false;
  MSLOG_IF(INFO, ret == 0, NoExceptionType) << "Event base dispatch success!";
  MSLOG_IF(mindspore::ERROR, ret == 1, NoExceptionType)
    << "Event base dispatch failed with no events pending or active!";
  MSLOG_IF(mindspore::ERROR, ret == -1, NoExceptionType) << "Event base dispatch failed with error occurred!";
  MSLOG_IF(mindspore::EXCEPTION, ret < -1, AbortedError) << "Event base dispatch with unexpected error code!";
}

void TcpClient::StartWithNoBlock() {
  std::lock_guard<std::mutex> lock(connection_mutex_);
  MS_LOG(INFO) << "Start tcp client with no block!";
  MS_EXCEPTION_IF_NULL(event_base_);
  int ret = event_base_loop(event_base_, EVLOOP_NONBLOCK);
  MSLOG_IF(INFO, ret == 0, NoExceptionType) << "Event base loop success!";
  MSLOG_IF(mindspore::ERROR, ret == 1, NoExceptionType) << "Event base loop failed with no events pending or active!";
  MSLOG_IF(mindspore::ERROR, ret == -1, NoExceptionType) << "Event base loop failed with error occurred!";
  MSLOG_IF(mindspore::EXCEPTION, ret < -1, AbortedError) << "Event base loop with unexpected error code!";
}

void TcpClient::SetMessageCallback(const OnMessage &cb) { message_callback_ = cb; }

bool TcpClient::SendMessage(const CommMessage &message) const {
  MS_EXCEPTION_IF_NULL(buffer_event_);
  bufferevent_lock(buffer_event_);
  bool res = true;
  size_t buf_size = message.ByteSizeLong();
  uint32_t meta_size = SizeToUint(message.pb_meta().ByteSizeLong());
  MessageHeader header;
  header.message_proto_ = Protos::PROTOBUF;
  header.message_length_ = buf_size;
  header.message_meta_length_ = meta_size;
  if (bufferevent_write(buffer_event_, &header, sizeof(header)) == -1) {
    MS_LOG(ERROR) << "Event buffer add header failed!";
    res = false;
  }
  if (bufferevent_write(buffer_event_, message.pb_meta().SerializeAsString().data(), meta_size) == -1) {
    MS_LOG(ERROR) << "Event buffer add protobuf data failed!";
    res = false;
  }
  if (bufferevent_write(buffer_event_, message.data().data(), message.data().length()) == -1) {
    MS_LOG(ERROR) << "Event buffer add protobuf data failed!";
    res = false;
  }
  bufferevent_unlock(buffer_event_);
  return res;
}

bool TcpClient::SendMessage(const std::shared_ptr<MessageMeta> &meta, const Protos &protos, const void *data,
                            size_t size) {
  MS_EXCEPTION_IF_NULL(buffer_event_);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  bufferevent_lock(buffer_event_);
  bool res = true;

  MessageHeader header;
  header.message_proto_ = protos;
  header.message_meta_length_ = SizeToUint(meta->ByteSizeLong());
  header.message_length_ = size + header.message_meta_length_;

  if (bufferevent_write(buffer_event_, &header, sizeof(header)) == -1) {
    MS_LOG(ERROR) << "Event buffer add header failed!";
    res = false;
  }
  if (bufferevent_write(buffer_event_, meta->SerializeAsString().data(), meta->ByteSizeLong()) == -1) {
    MS_LOG(ERROR) << "Event buffer add protobuf data failed!";
    res = false;
  }
  if (bufferevent_write(buffer_event_, data, size) == -1) {
    MS_LOG(ERROR) << "Event buffer add protobuf data failed!";
    res = false;
  }
  int result = bufferevent_flush(buffer_event_, EV_READ | EV_WRITE, BEV_FLUSH);
  if (result < 0) {
    MS_LOG(ERROR) << "Bufferevent flush failed!";
    res = false;
  }
  bufferevent_unlock(buffer_event_);
  return res;
}

void TcpClient::set_timer_callback(const OnTimer &timer) { on_timer_callback_ = timer; }

const event_base &TcpClient::eventbase() const { return *event_base_; }
}  // namespace core
}  // namespace ps
}  // namespace mindspore
