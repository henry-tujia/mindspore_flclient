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

#include <memory>
#include "tools/optimizer/fusion/activation_fusion.h"
#include "test/ut/tools/optimizer/fusion/fusion_inout_test/fusion_inout_test.h"
#include "backend/kernel_compiler/cpu/nnacl/op_base.h"
#include "ops/fusion/activation.h"

namespace mindspore {
namespace {
inline const int kActMinVal = -20;
inline const int kActMaxVal = 6;
}  // namespace
class ActivationFusionInoutTest : public FusionInoutTest {
 public:
  ActivationFusionInoutTest() = default;

 protected:
  void InitPass() override { this->pass_ = std::make_shared<opt::ActivationFusion>(); }

  void InitGraph() override {
    this->graph_ = std::make_shared<FuncGraph>();
    MS_CHECK_TRUE_MSG(graph_ != nullptr, , "Create FuncGraph failed");
    auto input = AddParameter(graph_, 0, {1, C16NUM, C16NUM, C3NUM}, kNumberTypeFloat32, "graph_input");
    if (input == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto act_1 = AddAct(graph_, input, ActivationType::HARD_TANH, "act_1", kActMinVal, kActMaxVal);
    if (act_1 == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto act_2 = AddAct(graph_, act_1, ActivationType::RELU, "act_2");
    if (act_2 == nullptr) {
      this->graph_ = nullptr;
      return;
    }
    auto ret = AddReturn(graph_, {act_2});
    if (ret == nullptr) {
      this->graph_ = nullptr;
      return;
    }
  }

 private:
  static CNodePtr AddAct(const FuncGraphPtr &graph, const AnfNodePtr &input, const ActivationType &type,
                         const std::string &name, const float &min_val = FLT_MAX, const float &max_val = -FLT_MAX) {
    auto prim = std::make_unique<ops::Activation>();
    MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "create Act primitivec failed");
    prim->Init();
    prim->set_activation_type(type);
    if (type == ActivationType::HARD_TANH) {
      prim->set_min_val(min_val);
      prim->set_max_val(max_val);
    }
    auto act_primitive = NewValueNode(std::shared_ptr<ops::PrimitiveC>(prim.release()));
    MS_CHECK_TRUE_RET(act_primitive != nullptr, nullptr);
    auto act = graph->NewCNode({act_primitive, input});
    MS_CHECK_TRUE_MSG(act != nullptr, nullptr, "create Act failed");
    act->set_fullname_with_scope(name);
    return act;
  }
};

TEST_F(ActivationFusionInoutTest, test) { ASSERT_EQ(DoTest(), true); }
}  // namespace mindspore
