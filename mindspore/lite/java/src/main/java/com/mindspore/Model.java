/*
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

package com.mindspore;

import com.mindspore.config.ModelType;
import com.mindspore.config.MSContext;
import com.mindspore.config.TrainCfg;

import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class Model {
    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private long modelPtr = 0;

    /**
     * Construct function.
     */
    public Model() {
        this.modelPtr = 0;
    }

    /**
     * Build model by graph.
     *
     * @param graph   graph contains the buffer.
     * @param context model build context.
     * @param cfg     model build train config.used for train.
     * @return build status.
     */
    public boolean build(Graph graph, MSContext context, TrainCfg cfg) {
        if (graph == null || context == null) {
            return false;
        }
        long cfgPtr = cfg != null ? cfg.getTrainCfgPtr() : 0;
        modelPtr = this.buildByGraph(graph.getGraphPtr(), context.getMSContextPtr(), cfgPtr);
        return modelPtr != 0;
    }

    /**
     * Build model.
     *
     * @param buffer    model buffer.
     * @param modelType model type.
     * @param context   model build context.
     * @param dec_key   define the key used to decrypt the ciphertext model. The key length is 16, 24, or 32.
     * @param dec_mode  define the decryption mode. Options: AES-GCM, AES-CBC.
     * @return model build status.
     */
    public boolean build(final MappedByteBuffer buffer, int modelType, MSContext context, char[] dec_key,
                         String dec_mode) {
        if (context == null) {
            return false;
        }
        modelPtr = this.buildByBuffer(buffer, modelType, context.getMSContextPtr(), dec_key, dec_mode);
        return modelPtr != 0;
    }

    /**
     * Build model.
     *
     * @param buffer    model buffer.
     * @param modelType model type.
     * @param context   model build context.
     * @return model build status.
     */
    public boolean build(final MappedByteBuffer buffer, int modelType, MSContext context) {
        if (context == null) {
            return false;
        }
        modelPtr = this.buildByBuffer(buffer, modelType, context.getMSContextPtr(), null, "");
        return modelPtr != 0;
    }


    /**
     * Build model.
     *
     * @param modelPath model path.
     * @param modelType model type.
     * @param context   model build context.
     * @param dec_key   define the key used to decrypt the ciphertext model. The key length is 16, 24, or 32.
     * @param dec_mode  define the decryption mode. Options: AES-GCM, AES-CBC.
     * @return model build status.
     */
    public boolean build(String modelPath, int modelType, MSContext context, char[] dec_key, String dec_mode) {
        if (context == null) {
            return false;
        }
        modelPtr = this.buildByPath(modelPath, modelType, context.getMSContextPtr(), dec_key, dec_mode);
        return modelPtr != 0;
    }

    /**
     * Build model.
     *
     * @param modelPath model path.
     * @param modelType model type.
     * @param context   model build context.
     * @return build status.
     */
    public boolean build(String modelPath, int modelType, MSContext context) {
        if (context == null) {
            return false;
        }
        modelPtr = this.buildByPath(modelPath, modelType, context.getMSContextPtr(), null, "");
        return modelPtr != 0;
    }

    /**
     * Execute predict.
     *
     * @return predict status.
     */
    public boolean predict() {
        return this.runStep(modelPtr);
    }

    /**
     * Run Model by step.
     *
     * @return run model status.work in train mode.
     */
    public boolean runStep() {
        return this.runStep(modelPtr);
    }

    /**
     * Resize inputs shape.
     *
     * @param inputs Model inputs.
     * @param dims   Define the new inputs shape.
     * @return Whether the resize is successful.
     */
    public boolean resize(List<MSTensor> inputs, int[][] dims) {
        if (inputs == null || dims == null) {
            return false;
        }
        long[] inputsArray = new long[inputs.size()];
        for (int i = 0; i < inputs.size(); i++) {
            inputsArray[i] = inputs.get(i).getMSTensorPtr();
        }
        return this.resize(this.modelPtr, inputsArray, dims);
    }

    /**
     * Get model inputs tensor.
     *
     * @return input tensors.
     */
    public List<MSTensor> getInputs() {
        List<Long> ret = this.getInputs(this.modelPtr);
        List<MSTensor> tensors = new ArrayList<>();
        for (Long msTensorAddr : ret) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        return tensors;
    }

    /**
     * Get model outputs.
     *
     * @return model outputs tensor.
     */
    public List<MSTensor> getOutputs() {
        List<Long> ret = this.getOutputs(this.modelPtr);
        List<MSTensor> tensors = new ArrayList<>();
        for (Long msTensorAddr : ret) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        return tensors;
    }

    /**
     * Get input tensor by tensor name.
     *
     * @param tensorName name.
     * @return input tensor.
     */
    public MSTensor getInputByTensorName(String tensorName) {
        long tensorAddr = this.getInputByTensorName(this.modelPtr, tensorName);
        return new MSTensor(tensorAddr);
    }

    /**
     * Get output tensor by tensor name.
     *
     * @param tensorName output tensor name
     * @return output tensor
     */
    public MSTensor getOutputByTensorName(String tensorName) {
        long tensorAddr = this.getOutputByTensorName(this.modelPtr, tensorName);
        return new MSTensor(tensorAddr);
    }

    /**
     * Export the model.
     *
     * @param fileName          Name Model file name.
     * @param quantizationType  The quant type.0,no_quant,1,weight_quant,2,full_quant.
     * @param isOnlyExportInfer if export only inferece.
     * @param outputTensorNames tensor name used for export inference graph.
     * @return Whether the export is successful.
     */
    public boolean export(String fileName, int quantizationType, boolean isOnlyExportInfer,
                          List<String> outputTensorNames) {
        return export(modelPtr, fileName, quantizationType, isOnlyExportInfer, outputTensorNames);
    }

    /**
     * Get the FeatureMap.
     *
     * @return FeaturesMap Tensor list.
     */
    public List<MSTensor> getFeatureMaps() {
        List<Long> ret = this.getFeatureMaps(this.modelPtr);
        ArrayList<MSTensor> tensors = new ArrayList<>();
        for (Long msTensorAddr : ret) {
            MSTensor msTensor = new MSTensor(msTensorAddr);
            tensors.add(msTensor);
        }
        return tensors;
    }

    /**
     * Update model Features.
     *
     * @param features new FeatureMap Tensor List.
     * @return Whether the model features is successfully update.
     */
    public boolean updateFeatureMaps(List<MSTensor> features) {
        if (features == null) {
            return false;
        }
        long[] inputsArray = new long[features.size()];
        for (int i = 0; i < features.size(); i++) {
            inputsArray[i] = features.get(i).getMSTensorPtr();
        }
        return this.updateFeatureMaps(modelPtr, inputsArray);
    }

    /**
     * Set model work train mode
     *
     * @param isTrain is train mode.true work train mode.
     * @return set status.
     */
    public boolean setTrainMode(boolean isTrain) {
        return this.setTrainMode(modelPtr, isTrain);
    }

    /**
     * Get train mode
     *
     * @return train mode.
     */
    public boolean getTrainMode() {
        return this.getTrainMode(modelPtr);
    }

    /**
     * Free model
     */
    public void free() {
        this.free(modelPtr);
    }

    private native void free(long modelPtr);

    private native long buildByGraph(long graphPtr, long contextPtr, long cfgPtr);

    private native long buildByPath(String modelPath, int modelType, long contextPtr, char[] dec_key, String dec_mod);

    private native long buildByBuffer(MappedByteBuffer buffer, int modelType, long contextPtr, char[] dec_key,
                                      String dec_mod);

    private native List<Long> getInputs(long modelPtr);

    private native long getInputByTensorName(long modelPtr, String tensorName);

    private native boolean runStep(long modelPtr);

    private native List<Long> getOutputs(long modelPtr);

    private native long getOutputByTensorName(long modelPtr, String tensorName);

    private native boolean setTrainMode(long modelPtr, boolean isTrain);

    private native boolean getTrainMode(long modelPtr);

    private native boolean resize(long modelPtr, long[] inputs, int[][] dims);

    private native boolean export(long modelPtr, String fileName, int quantizationType, boolean isOnlyExportInfer,
                                  List<String> outputTensorNames);

    private native List<Long> getFeatureMaps(long modelPtr);

    private native boolean updateFeatureMaps(long modelPtr, long[] newFeatures);
}
