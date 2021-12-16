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

import com.alibaba.fastjson.JSONObject;

import java.io.*;
import java.lang.reflect.Type;
import java.util.*;

public class Dataset {

    public static class DataLabelTuple {
        public ArrayList<Float> feat_vals;
        public ArrayList<Float> label;
        public ArrayList<Integer> feat_ids;
    }

    private Vector<DataLabelTuple> trainData;
    private Vector<DataLabelTuple> testData;

    public void initDataset(String InputFileName, Boolean trainFlag) {
        trainData = new Vector<>();
        testData = new Vector<>();
        if (trainFlag) {
            readfile(InputFileName, trainData);
        } else {
            readfile(InputFileName, testData);
        }
    }

    private void readfile(String InputFileName, Vector<DataLabelTuple> dataset) {
        try {
            File file = new File(InputFileName);
            BufferedReader br = new BufferedReader(new FileReader(file));
            String st;
            int count = 0;
            while ((st = br.readLine()) != null) {
                System.out.println(count+"th's data reading!");
                JSONObject object = JSONObject.parseObject(st);
                DataLabelTuple dataLabelTuple = object.toJavaObject(DataLabelTuple.class);
//                DataLabelTuple dataLabelTuple = new DataLabelTuple();
//                JSONArray feat_ids = (JSONArray) object.get("feat_ids");
//                dataLabelTuple.feat_ids = (ArrayList<Integer>) JSONObject.parseArray(feat_ids.toJSONString(), Integer.class);
//                JSONArray feat_vals = (JSONArray) object.get("feat_vals");
//                dataLabelTuple.feat_vals = (ArrayList<Float>) JSONObject.parseArray(feat_ids.toJSONString(), Float.class);
//                JSONArray label = (JSONArray) object.get("label");
//                dataLabelTuple.label = Float.parseFloat(label.get(0).toString());
                dataset.add(dataLabelTuple);
                count ++;
            }
        } catch (IOException s) {
            s.printStackTrace();
        }
    }

    public Vector<DataLabelTuple> getTrainData() {
        return trainData;
    }

    public Vector<DataLabelTuple> getTestData() {
        return testData;
    }
}
