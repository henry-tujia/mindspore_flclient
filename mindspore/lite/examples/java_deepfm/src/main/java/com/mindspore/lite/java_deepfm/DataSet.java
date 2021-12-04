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

package com.mindspore.lite.java_deepfm;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Vector;

public class DataSet {
    private long numOfClasses = 0;
    private long expectedDataSize = 0;
    public class DataLabelTuple {
        public int[] ids;
        public float[] vals;
        public int label;
    }
    Vector<DataLabelTuple> trainData;
    Vector<DataLabelTuple> testData;

    public void initializeCriteoDatabase(String dpath) {
        numOfClasses = 10;
        trainData = new Vector<DataLabelTuple>();
        testData = new Vector<DataLabelTuple>();
        readMNISTFile(dpath + "/train/train-images-idx3-ubyte", dpath+"/train/train-labels-idx1-ubyte", trainData);
        readMNISTFile(dpath + "/test/t10k-images-idx3-ubyte", dpath+"/test/t10k-labels-idx1-ubyte", testData);

        System.out.println("train data cnt: " + trainData.size());
        System.out.println("test data cnt: " + testData.size());
    }

    private String bytesToHex(byte[] bytes) {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < bytes.length; i++) {
            String hex = Integer.toHexString(bytes[i] & 0xFF);
            if (hex.length() < 2) {
                sb.append(0);
            }
            sb.append(hex);
        }
        return sb.toString();
    }

    private void readFile(BufferedInputStream inputStream, byte[] bytes, int len) throws IOException {
        int result = inputStream.read(bytes, 0, len);
        if (result != len) {
            System.err.println("expected read " + len + " bytes, but " + result + " read");
            System.exit(1);
        }
    }
    public void readCriteoFile(String inputFileName, String labelFileName, Vector<DataLabelTuple> dataset) {
        try {
            BufferedInputStream ibin = new BufferedInputStream(new FileInputStream(inputFileName));
            byte[] bytes = new byte[4];

            readFile(ibin, bytes, 4);
            if (!"00000803".equals(bytesToHex(bytes))) { // 2051
                System.err.println("The dataset is not valid: " + bytesToHex(bytes));
                return;
            }
            readFile(ibin, bytes, 4);
            int inumber = Integer.parseInt(bytesToHex(bytes), 16);

            // row, column
            readFile(ibin, bytes, 4);
            int n_rows = Integer.parseInt(bytesToHex(bytes), 16);
            readFile(ibin, bytes, 4);
            int n_cols = Integer.parseInt(bytesToHex(bytes), 16);
            if (n_rows != 28 || n_cols != 28) {
                System.err.println("invalid  n_rows: " + n_rows + " n_cols: " + n_cols);
                return;
            }
            // read images
            int image_size = n_rows * n_cols;
            byte[] image_data = new byte[image_size];

        } catch (IOException e) {
            System.err.println("Read Dateset exception");
        }
    }

    public void setExpectedDataSize(long data_size) {
        expectedDataSize = data_size;
    }

    public long getNumOfClasses() {
        return numOfClasses;
    }

    public Vector<DataLabelTuple> getTestData() {
        return testData;
    }

    public Vector<DataLabelTuple> getTrainData() {
        return trainData;
    }
}
