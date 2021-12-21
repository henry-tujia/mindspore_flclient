/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_CONNECTION_POOL_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_CONNECTION_POOL_H_

#include <map>
#include <set>
#include <string>

#include "distributed/rpc/tcp/constants.h"
#include "distributed/rpc/tcp/connection.h"

namespace mindspore {
namespace distributed {
namespace rpc {
struct ConnectionInfo {
  int socket_fd;
  std::string from;
  std::string to;
  DeleteCallBack delete_callback;
};

/*
 * Maintains a collection of reusable connections.
 */
class ConnectionPool {
 public:
  ConnectionPool() : double_link_(false) {}
  ~ConnectionPool();

  // Get the singleton instance of ConnectionPool.
  static ConnectionPool *GetConnectionPool();

  /*
   * Operations for ConnectionInfo.
   */
  void AddConnInfo(int socket_fd, const AID &sAid, const AID &dAid, DeleteCallBack delcb);
  bool ReverseConnInfo(int from_socket_fd, int to_socket_fd);

  /*
   * Operations for Connection.
   */
  // Add a connection.
  void AddConnection(Connection *conn);

  // Find connection.
  Connection *ExactFindConnection(const std::string &to, bool remoteLink);
  Connection *FindConnection(const std::string &to, bool remoteLink);
  Connection *FindConnection(const std::string &to, bool remoteLink, bool exactNotRemote);
  Connection *FindMaxConnection();
  Connection *FindFastConnection();

  // Delete connection.
  void ExactDeleteConnection(const std::string &to, bool remoteLink);
  void DeleteAllConnections(std::map<std::string, Connection *> *alllinks);

  // Close connection.
  void CloseConnection(Connection *conn);
  void SetConnPriority(const std::string &to, bool remoteLink, ConnectionPriority pri);

  // Single link or double link.
  void SetLinkPattern(bool linkPattern);

  void ResetAllConnMetrics();

 private:
  ConnectionInfo *FindConnInfo(int socket_fd, const AID &sAid, const AID &dAid);

  void DeleteConnInfo(int socket_fd);
  void DeleteConnInfo(const std::string &to, int socket_fd);
  void DeleteAllConnInfos();

  bool double_link_;

  // to_url=tcp@ip:port, event struct
  std::map<std::string, Connection *> local_conns_;

  // Maintains the remote connections by remote server addresses.
  std::map<std::string, Connection *> remote_conns_;

  // each to_url has two fds at most, and each fd has multiple linkinfos
  std::map<int, std::set<ConnectionInfo *>> conn_infos_;

  static ConnectionPool *conn_pool;

  friend class Connection;
  friend class TCPComm;
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
#endif
