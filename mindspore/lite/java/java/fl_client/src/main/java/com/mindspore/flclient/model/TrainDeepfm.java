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
import com.mindspore.lite.LiteSession;
import com.mindspore.lite.MSTensor;

import java.util.Vector;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.logging.Logger;

public class TrainDeepfm extends TrainModel {
    private static final Logger logger = Logger.getLogger(TrainDeepfm.class.toString());

    private static final int NUM_OF_CLASS = 1;

    private DatasetDeepfm mDs = new DatasetDeepfm();

    private Vector<DatasetDeepfm.DataLabelTuple> dataset;

    // private Vector<DatasetDeepfm.DataLabelTuple> mTestDataset;

    private int batch_size;

    private int imageSize;

    private int labelSize;

    private float[] idsArray;

    private float[] valsArray;

    private int[] labelArray;

    private ByteBuffer labelsBuffer;

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

    // public int[] inferModel(String modelPath, String testFile) {
    //     if (modelPath.isEmpty() || testFile.isEmpty()) {
    //         logger.severe(Common.addTag("model path or image file cannot be empty"));
    //         return new int[0];
    //     }
    //     int trainSize = initDataSet(testFile, "");
    //     logger.info(Common.addTag("dataset origin size:" + trainSize));
    //     int status = initSessionAndInputs(modelPath, false);
    //     if (status == -1) {
    //         logger.severe(Common.addTag("init session and inputs failed"));
    //         return new int[0];
    //     }
    //     status = padSamples();
    //     if (status == -1) {
    //         logger.severe(Common.addTag("infer model failed"));
    //         return new int[0];
    //     }
    //     int[] predictLabels = new int[trainSampleSize];
    //     for (int j = 0; j < batchNum; j++) {
    //         fillModelInput(j, false);
    //         boolean success = trainSession.runGraph();
    //         if (!success) {
    //             logger.severe(Common.addTag("run graph failed"));
    //             return new int[0];
    //         }
    //         int[] batchLabels = getBatchLabel();
    //         System.arraycopy(batchLabels, 0, predictLabels, j * batchSize, batchSize);
    //     }
    //     if (predictLabels.length == 0) {
    //         return new int[0];
    //     }
    //     return Arrays.copyOfRange(predictLabels, 0, trainSampleSize - padSize);
    // }

    public int initDataSet(String inputFile,boolean Train) {
        if (!inputFile.isEmpty()) {
            mDs.initDataset(inputFile,Train);
            if (Train){
                dataset = mDs.getTrainData();
            }
            else{
                dataset = mDs.getTestData();
            }
        } else {
            return -1;  // labelArray may be initialized from train
        }
        return dataset.size();
    }


    @Override
    public int padSamples() {
        return 0;
    }

    @Override
    public int initSessionAndInputs(String modelPath, boolean trainMod) {
        if (modelPath == null) {
            logger.severe(Common.addTag("modelPath cannot be empty"));
            return -1;
        }
        Optional<LiteSession> optTrainSession = SessionUtil.initSession(modelPath);
        if (!optTrainSession.isPresent()) {
            logger.severe(Common.addTag("session init failed"));
            return -1;
        }
        trainSession = optTrainSession.get();
        List<MSTensor> inputs = trainSession.getInputs();
        batch_size = inputs.get(0).getShape()[0];
        batchNum = (int)Math.toIntExact(inputs.get(0).size()/batch_size);
        logger.info(Common.addTag("init session and inputs success"));
        return 0;
    }

    @Override
    public Vector<Integer> fillModelInput(int batchIdx, boolean trainMod) {
        labelsBuffer.clear();
        idsBuffer.clear();
        valsBuffer.clear();
        List<Integer> predictLabels = new ArrayList<>(batch_size);
        Vector<Integer> labelsVec = new Vector<>();

        
        List<MSTensor> inputs = trainSession.getInputs();

        int inputIdsDataCnt = inputs.get(0).elementsNum();
        int[] inputIdsBatchData = new int[inputIdsDataCnt];

        int inputValSDataCnt = inputs.get(1).elementsNum();
        float[] inputValsBatchData = new float[inputValSDataCnt];

        int labelDataCnt = inputs.get(2).elementsNum();
        int[] labelBatchData = new int[labelDataCnt];

        logger.info(Common.addTag("total batchNum:" + batchSize));

        for (int i = 0; i < batchSize; i++) {
            DatasetDeepfm.DataLabelTuple dataLabelTuple = dataset.get(batchIdx*batch_size+i);
            int label = dataLabelTuple.label.get(0).intValue();
            int[] ids = dataLabelTuple.feat_ids.stream().mapToInt(j -> j).toArray();

            int n = 0;
            float[] vals = new float[dataLabelTuple.feat_vals.size()];
            for (Float f : dataLabelTuple.feat_vals) {
                vals[n++] = (f != null ? f : Float.NaN); // Or whatever default you want.
            }
            System.arraycopy(ids, 0, inputIdsBatchData, i * ids.length, ids.length);
            System.arraycopy(vals, 0, inputValsBatchData, i * vals.length, vals.length);
            labelBatchData[i] = label;
            labelsVec.add(label);
        }

        ByteBuffer byteBufIds = ByteBuffer.allocateDirect(inputIdsBatchData.length * Integer.BYTES);
        byteBufIds.order(ByteOrder.nativeOrder());
        for (int i = 0; i < inputIdsBatchData.length; i++) {
            byteBufIds.putFloat(inputIdsBatchData[i]);
        }
        inputs.get(0).setData(byteBufIds);

        ByteBuffer byteBufVals = ByteBuffer.allocateDirect(inputValsBatchData.length * Float.BYTES);
        byteBufVals.order(ByteOrder.nativeOrder());
        for (int i = 0; i < inputIdsBatchData.length; i++) {
            byteBufVals.putFloat(inputIdsBatchData[i]);
        }
        inputs.get(1).setData(byteBufVals);

        ByteBuffer labelByteBuf = ByteBuffer.allocateDirect(labelBatchData.length * 4);
        labelByteBuf.order(ByteOrder.nativeOrder());
        for (int i = 0; i < labelBatchData.length; i++) {
            labelByteBuf.putInt(labelBatchData[i]);
        }
        inputs.get(2).setData(labelByteBuf);
        return labelsVec;
    }

    
    public int getPredictLabel(float[] scores, int start, int end) {
        if (scores == null || scores.length == 0) {
            logger.severe(Common.addTag("scores cannot be empty"));
            return -1;
        }
        if (start >= scores.length || start < 0 || end > scores.length || end < 0) {
            logger.severe(Common.addTag("start,end cannot out of scores length"));
            return -1;
        }
        if (end-start>1 ) {
            logger.severe(Common.addTag("the diff between end and start cannot out of 1"));
            return -1;
        }
        float maxScore = scores[start];
        // int maxIdx = start;
        // for (int i = start; i < end; i++) {
        //     if (scores[i] > maxScore) {
        //         maxIdx = i;
        //         maxScore = scores[i];
        //     }
        // }
        return Math.round(1 / (1 + (float) Math.exp(-maxScore)));
    }
}
