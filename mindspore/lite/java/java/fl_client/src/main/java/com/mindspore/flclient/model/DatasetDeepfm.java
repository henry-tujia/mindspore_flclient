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

package com.mindspore.flclient.model;

import com.alibaba.fastjson.JSONObject;

import java.io.*;
import java.lang.reflect.Type;
import java.util.*;
import java.util.logging.Logger;

public class DatasetDeepfm {
    private static final Logger logger = Logger.getLogger(DatasetDeepfm.class.toString());

    public static class DataLabelTuple {
        public ArrayList<Float> feat_vals;
        public ArrayList<Float> label;
        public ArrayList<Integer> feat_ids;
    }

    private Vector<DataLabelTuple> trainData;
    private Vector<DataLabelTuple> testData;

    public void initDataset(String InputName,boolean Train) {
        if (Train){
            trainData = new Vector<>();
            readfile(InputName, trainData);
        }
    else{
            testData = new Vector<>();
            readfile(InputName, testData);
        }   
    }

    private void readfile(String InputFileName, Vector<DataLabelTuple> dataset) {
        try {
            File file = new File(InputFileName);
            BufferedReader br = new BufferedReader(new FileReader(file));
            String st;
            int count = 0;
            while ((st = br.readLine()) != null) {
                if (count>999){
                    break;
                }
                // System.out.println(count+"th's data reading!");
                DataLabelTuple dataLabelTuple = JSONObject.parseObject(st,DataLabelTuple.class);
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
