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

import com.mindspore.flclient.Common;
import com.mindspore.lite.MSTensor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

public class TrainDeepfm extends TrainModel {
    private static final Logger logger = Logger.getLogger(TrainDeepfm.class.toString());

    private static final int NUM_OF_CLASS = 62;

    private DatasetDeepfm dataset;

    private int imageSize;

    private int labelSize;

    private float[] idsArray;

    private float[] valsArray;

    private int[] labelArray;

    private ByteBuffer labelIdBuffer;

    private ByteBuffer idsBuffer;

    private ByteBuffer valsBuffer;

    private static volatile TrainDeepfm trainDeepfm;

    public static TrainDeepfm getInstance() {
        TrainDeepfm localRef = trainDeepfm;
        if (localRef == null) {
            synchronized (TrainDeepfm.class) {
                localRef = trainDeepfm;
                if (localRef == null) {
                    trainDeepfm = localRef = new TrainDeepfm();
                }
            }
        }
        return localRef;
    }

    public int[] inferModel(String modelPath, String testFile) {
        if (modelPath.isEmpty() || testFile.isEmpty()) {
            logger.severe(Common.addTag("model path or image file cannot be empty"));
            return new int[0];
        }
        int trainSize = initDataSet(testFile, "");
        logger.info(Common.addTag("dataset origin size:" + trainSize));
        int status = initSessionAndInputs(modelPath, false);
        if (status == -1) {
            logger.severe(Common.addTag("init session and inputs failed"));
            return new int[0];
        }
        status = padSamples();
        if (status == -1) {
            logger.severe(Common.addTag("infer model failed"));
            return new int[0];
        }
        int[] predictLabels = new int[trainSampleSize];
        for (int j = 0; j < batchNum; j++) {
            fillModelInput(j, false);
            boolean success = trainSession.runGraph();
            if (!success) {
                logger.severe(Common.addTag("run graph failed"));
                return new int[0];
            }
            int[] batchLabels = getBatchLabel();
            System.arraycopy(batchLabels, 0, predictLabels, j * batchSize, batchSize);
        }
        if (predictLabels.length == 0) {
            return new int[0];
        }
        return Arrays.copyOfRange(predictLabels, 0, trainSampleSize - padSize);
    }

    public int initDataSet(String inputFile,boolean Train) {
        if (!inputFile.isEmpty()) {
            dataset.initDataset(inputFile,Train);
            trainDataset = dataset.getTrainData();
            ArrayList<Float> label = trainDataset.label;
            // byte[] labelByteArray = DataSet.readBinFile(labelFile);
            int trainSize = label.length;
            trainSampleSize = trainSize;
            // label is 32,need pad 32*62
            labelArray = new int[trainSize * NUM_OF_CLASS];
            Arrays.fill(labelArray, 0);
            int j = 0;
            for (int i = 0; i < labelByteArray.length; i += 4) {
                labelArray[j * NUM_OF_CLASS + labelByteArray[i]] = 1;
                j++;
            }
        } else {
            labelArray = null;  // labelArray may be initialized from train
        }

        return trainSampleSize;
    }

    @Override
    public int initSessionAndInputs(String modelPath, boolean trainMod) {
        if (modelPath.isEmpty()) {
            logger.severe(Common.addTag("modelPath cannot be empty"));
            return -1;
        }
        trainSession = SessionUtil.initSession(modelPath);
        if (trainSession == null) {
            logger.severe(Common.addTag("session init failed"));
            return -1;
        }
        numOfClass = NUM_OF_CLASS;
        List<MSTensor> inputs = trainSession.getInputs();
        MSTensor imageTensor = inputs.get(0);
        batchSize = imageTensor.getShape()[0];
        imageSize = imageTensor.elementsNum();
        imageBuffer = ByteBuffer.allocateDirect(imageSize * Float.BYTES);
        imageBuffer.order(ByteOrder.nativeOrder());
        MSTensor labelTensor = inputs.get(1);
        labelSize = labelTensor.elementsNum();
        labelIdBuffer = ByteBuffer.allocateDirect(labelSize * Integer.BYTES);
        labelIdBuffer.order(ByteOrder.nativeOrder());
        return 0;
    }

    @Override
    public List<Integer> fillModelInput(int batchIdx, boolean trainMod) {
        imageBuffer.clear();
        labelIdBuffer.clear();
        List<Integer> predictLabels = new ArrayList<>(batchSize);
        for (int i = 0; i < imageSize * Float.BYTES; i++) {
            imageBuffer.put(imageArray[batchIdx * imageSize * Float.BYTES + i]);
        }
        for (int i = 0; i < labelSize; i++) {
            labelIdBuffer.putFloat(labelArray[batchIdx * labelSize + i]);
            if (!trainMod && labelArray[batchIdx * labelSize + i] == 1) {
                predictLabels.add(i % NUM_OF_CLASS);
            }
        }

        List<MSTensor> inputs = trainSession.getInputs();
        MSTensor imageTensor = inputs.get(0);
        MSTensor labelTensor = inputs.get(1);
        imageTensor.setData(imageBuffer);
        labelTensor.setData(labelIdBuffer);
        return predictLabels;
    }
}
