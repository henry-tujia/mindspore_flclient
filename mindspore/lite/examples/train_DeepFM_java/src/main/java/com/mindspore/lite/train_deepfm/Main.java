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

package com.mindspore.lite.train_deepfm;

import com.mindspore.lite.Version;

public  class Main {
    public static void main(String[] args) {
        System.loadLibrary("mindspore-lite-jni");
        System.out.println(Version.version());

        String modelPath = "/mnt/data/th/deepFm_java/deepfm.ms";
        String datasetPath = "/mnt/data/th/deepFm_java/";
        String virtualBatch = "16";

        NetRunner net_runner = new NetRunner();
        net_runner.trainModel(modelPath, datasetPath, Integer.parseInt(virtualBatch));
    }


}
