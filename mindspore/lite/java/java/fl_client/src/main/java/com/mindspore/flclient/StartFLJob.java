/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
 */

package com.mindspore.flclient;

import com.google.flatbuffers.FlatBufferBuilder;

import com.mindspore.flclient.model.AlInferBert;
import com.mindspore.flclient.model.AlTrainBert;
import com.mindspore.flclient.model.Client;
import com.mindspore.flclient.model.ClientManager;
import com.mindspore.flclient.model.RunType;
import com.mindspore.flclient.model.SessionUtil;
import com.mindspore.flclient.model.Status;
import com.mindspore.flclient.model.TrainLenet;
import com.mindspore.flclient.model.TrainDeepfm;
import com.mindspore.flclient.pki.PkiBean;
import com.mindspore.flclient.pki.PkiUtil;

import mindspore.schema.FLPlan;
import mindspore.schema.FeatureMap;
import mindspore.schema.RequestFLJob;
import mindspore.schema.ResponseCode;
import mindspore.schema.ResponseFLJob;

import java.io.IOException;
import java.security.cert.Certificate;
import java.util.ArrayList;
import java.util.logging.Logger;

import static com.mindspore.flclient.LocalFLParameter.ALBERT;
import static com.mindspore.flclient.LocalFLParameter.LENET;
import static com.mindspore.flclient.LocalFLParameter.DEEPFM;

/**
 * StartFLJob
 *
 * @since 2021-08-25
 */
public class StartFLJob {
    private static final Logger LOGGER = Logger.getLogger(StartFLJob.class.toString());
    private static volatile StartFLJob startFLJob;

    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();

    private int featureSize;
    private String nextRequestTime;
    private ArrayList<String> updateFeatureName = new ArrayList<String>();
    private int retCode = ResponseCode.RequestError;
    private float lr = (float) 0.1;
    private int batchSize;

    private StartFLJob() {
    }

    /**
     * getInstance of StartFLJob
     *
     * @return StartFLJob instance
     */
    public static StartFLJob getInstance() {
        StartFLJob localRef = startFLJob;
        if (localRef == null) {
            synchronized (StartFLJob.class) {
                localRef = startFLJob;
                if (localRef == null) {
                    startFLJob = localRef = new StartFLJob();
                }
            }
        }
        return localRef;
    }

    public String getNextRequestTime() {
        return nextRequestTime;
    }

    public int getRetCode() {
        return retCode;
    }

    /**
     * get request start FLJob
     *
     * @param dataSize  dataSize
     * @param iteration iteration
     * @param time      time
     * @param pkiBean   pki bean
     * @return byte[] data
     */
    public byte[] getRequestStartFLJob(int dataSize, int iteration, long time, PkiBean pkiBean) {
        RequestStartFLJobBuilder builder = new RequestStartFLJobBuilder();

        if (flParameter.isPkiVerify()) {
            if (pkiBean == null) {
                LOGGER.severe(Common.addTag("[startFLJob] the parameter of <pkiBean> is null, please check!"));
                throw new IllegalArgumentException();
            }
            return builder.flName(flParameter.getFlName())
                    .time(time)
                    .id(localFLParameter.getFlID())
                    .dataSize(dataSize)
                    .iteration(iteration)
                    .signData(pkiBean.getSignData())
                    .certificateChain(pkiBean.getCertificates())
                    .build();
        }
        return builder.flName(flParameter.getFlName())
                .time(time)
                .id(localFLParameter.getFlID())
                .dataSize(dataSize)
                .iteration(iteration)
                .build();
    }

    public int getFeatureSize() {
        return featureSize;
    }

    public ArrayList<String> getUpdateFeatureName() {
        return updateFeatureName;
    }

    private FLClientStatus deprecatedParseResponseAlbert(ResponseFLJob flJob) {
        FLClientStatus status;
        int fmCount = flJob.featureMapLength();
        updateFeatureName.clear();
        if (fmCount <= 0) {
            LOGGER.severe(Common.addTag("[startFLJob] the feature size get from server is zero"));
            return FLClientStatus.FAILED;
        }

        if (localFLParameter.getServerMod().equals(ServerMod.HYBRID_TRAINING.toString())) {
            LOGGER.info(Common.addTag("[startFLJob] parseResponseAlbert by " + localFLParameter.getServerMod()));
            ArrayList<FeatureMap> albertFeatureMaps = new ArrayList<FeatureMap>();
            ArrayList<FeatureMap> inferFeatureMaps = new ArrayList<FeatureMap>();
            featureSize = 0;
            for (int i = 0; i < fmCount; i++) {
                FeatureMap feature = flJob.featureMap(i);
                if (feature == null) {
                    LOGGER.severe(Common.addTag("[startFLJob] the feature returned from server is null"));
                    return FLClientStatus.FAILED;
                }
                String featureName = feature.weightFullname();
                if (localFLParameter.getAlbertWeightName().contains(featureName)) {
                    albertFeatureMaps.add(feature);
                    inferFeatureMaps.add(feature);
                    featureSize += feature.dataLength();
                    updateFeatureName.add(feature.weightFullname());
                } else if (localFLParameter.getClassifierWeightName().contains(featureName)) {
                    inferFeatureMaps.add(feature);
                } else {
                    continue;
                }
                LOGGER.info(Common.addTag("[startFLJob] weightFullname: " + feature.weightFullname() + ", " +
                        "weightLength: " + feature.dataLength()));
            }
            status = Common.initSession(flParameter.getTrainModelPath());
            if (status == FLClientStatus.FAILED) {
                retCode = ResponseCode.RequestError;
                return status;
            }
            int tag = 0;
            LOGGER.info(Common.addTag("[startFLJob] ----------------loading weight into inference " +
                    "model-----------------"));
            AlInferBert alInferBert = AlInferBert.getInstance();
            tag = SessionUtil.updateFeatures(alInferBert.getTrainSession(), flParameter.getInferModelPath(),
                    inferFeatureMaps);
            if (tag == -1) {
                LOGGER.severe(Common.addTag("[startFLJob] unsolved error code in <SessionUtil.updateFeatures>"));
                return FLClientStatus.FAILED;
            }
            LOGGER.info(Common.addTag("[startFLJob] ----------------loading weight into train model-----------------"));
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            tag = SessionUtil.updateFeatures(alTrainBert.getTrainSession(), flParameter.getTrainModelPath(),
                    albertFeatureMaps);
            if (tag == -1) {
                LOGGER.severe(Common.addTag("[startFLJob] unsolved error code in <SessionUtil.updateFeatures>"));
                return FLClientStatus.FAILED;
            }
            Common.freeSession();
        } else if (localFLParameter.getServerMod().equals(ServerMod.FEDERATED_LEARNING.toString())) {
            LOGGER.info(Common.addTag("[startFLJob] parseResponseAlbert by " + localFLParameter.getServerMod()));
            ArrayList<FeatureMap> featureMaps = new ArrayList<FeatureMap>();
            featureSize = 0;
            for (int i = 0; i < fmCount; i++) {
                FeatureMap feature = flJob.featureMap(i);
                if (feature == null) {
                    LOGGER.severe(Common.addTag("[startFLJob] the feature returned from server is null"));
                    return FLClientStatus.FAILED;
                }
                String featureName = feature.weightFullname();
                featureMaps.add(feature);
                featureSize += feature.dataLength();
                updateFeatureName.add(featureName);
                LOGGER.info(Common.addTag("[startFLJob] weightFullname: " + feature.weightFullname() + ", " +
                        "weightLength: " + feature.dataLength()));
            }
            status = Common.initSession(flParameter.getTrainModelPath());
            if (status == FLClientStatus.FAILED) {
                retCode = ResponseCode.RequestError;
                return status;
            }
            int tag = 0;
            LOGGER.info(Common.addTag("[startFLJob] ----------------loading weight into model-----------------"));
            AlTrainBert alTrainBert = AlTrainBert.getInstance();
            tag = SessionUtil.updateFeatures(alTrainBert.getTrainSession(), flParameter.getTrainModelPath(),
                    featureMaps);
            if (tag == -1) {
                LOGGER.severe(Common.addTag("[startFLJob] unsolved error code in <SessionUtil.updateFeatures>"));
                return FLClientStatus.FAILED;
            }
            Common.freeSession();
        }
        return FLClientStatus.SUCCESS;
    }

    private FLClientStatus deprecatedParseResponseLenet(ResponseFLJob flJob) {
        FLClientStatus status;
        int fmCount = flJob.featureMapLength();
        ArrayList<FeatureMap> featureMaps = new ArrayList<FeatureMap>();
        updateFeatureName.clear();
        featureSize = 0;
        for (int i = 0; i < fmCount; i++) {
            FeatureMap feature = flJob.featureMap(i);
            if (feature == null) {
                LOGGER.severe(Common.addTag("[startFLJob] the feature returned from server is null"));
                return FLClientStatus.FAILED;
            }
            String featureName = feature.weightFullname();
            featureMaps.add(feature);
            featureSize += feature.dataLength();
            updateFeatureName.add(featureName);
            LOGGER.info(Common.addTag("[startFLJob] weightFullname: " +
                    feature.weightFullname() + ", weightLength: " + feature.dataLength()));
        }
        status = Common.initSession(flParameter.getTrainModelPath());
        if (status == FLClientStatus.FAILED) {
            retCode = ResponseCode.RequestError;
            return status;
        }
        int tag = 0;
        LOGGER.info(Common.addTag("[startFLJob] ----------------loading weight into model-----------------"));
        TrainLenet trainLenet = TrainLenet.getInstance();
        tag = SessionUtil.updateFeatures(trainLenet.getTrainSession(), flParameter.getTrainModelPath(), featureMaps);
        if (tag == -1) {
            LOGGER.severe(Common.addTag("[startFLJob] unsolved error code in <SessionUtil.updateFeatures>"));
            return FLClientStatus.FAILED;
        }
        Common.freeSession();
        return FLClientStatus.SUCCESS;
    }

    private FLClientStatus deprecatedParseResponseDeepfm(ResponseFLJob flJob) {
        FLClientStatus status;
        int fmCount = flJob.featureMapLength();
        ArrayList<FeatureMap> featureMaps = new ArrayList<FeatureMap>();
        updateFeatureName.clear();
        featureSize = 0;
        for (int i = 0; i < fmCount; i++) {
            FeatureMap feature = flJob.featureMap(i);
            if (feature == null) {
                LOGGER.severe(Common.addTag("[startFLJob] the feature returned from server is null"));
                return FLClientStatus.FAILED;
            }
            String featureName = feature.weightFullname();
            featureMaps.add(feature);
            featureSize += feature.dataLength();
            updateFeatureName.add(featureName);
            LOGGER.info(Common.addTag("[startFLJob] weightFullname: " +
                    feature.weightFullname() + ", weightLength: " + feature.dataLength()));
        }
        status = Common.initSession(flParameter.getTrainModelPath());
        if (status == FLClientStatus.FAILED) {
            retCode = ResponseCode.RequestError;
            return status;
        }
        int tag = 0;
        LOGGER.info(Common.addTag("[startFLJob] ----------------loading weight into model-----------------"));
        TrainDeepfm trainDeepfm = TrainDeepfm.getInstance();
        tag = SessionUtil.updateFeatures(trainDeepfm.getTrainSession(), flParameter.getTrainModelPath(), featureMaps);
        if (tag == -1) {
            LOGGER.severe(Common.addTag("[startFLJob] unsolved error code in <SessionUtil.updateFeatures>"));
            return FLClientStatus.FAILED;
        }
        Common.freeSession();
        return FLClientStatus.SUCCESS;
    }


    private FLClientStatus hybridFeatures(ResponseFLJob flJob) {
        FLClientStatus status;
        Client client = ClientManager.getClient(flParameter.getFlName());
        int fmCount = flJob.featureMapLength();
        ArrayList<FeatureMap> trainFeatureMaps = new ArrayList<FeatureMap>();
        ArrayList<FeatureMap> inferFeatureMaps = new ArrayList<FeatureMap>();
        featureSize = 0;
        for (int i = 0; i < fmCount; i++) {
            FeatureMap feature = flJob.featureMap(i);
            if (feature == null) {
                LOGGER.severe(Common.addTag("[startFLJob] the feature returned from server is null"));
                retCode = ResponseCode.SystemError;
                return FLClientStatus.FAILED;
            }
            String featureName = feature.weightFullname();
            if (flParameter.getHybridWeightName(RunType.TRAINMODE).contains(featureName)) {
                trainFeatureMaps.add(feature);
                featureSize += feature.dataLength();
                updateFeatureName.add(feature.weightFullname());
                LOGGER.info(Common.addTag("[startFLJob] trainWeightFullname: " + feature.weightFullname() + ", " +
                        "trainWeightLength: " + feature.dataLength()));
            }
            if (flParameter.getHybridWeightName(RunType.INFERMODE).contains(featureName)) {
                inferFeatureMaps.add(feature);
                LOGGER.info(Common.addTag("[startFLJob] inferWeightFullname: " + feature.weightFullname() + ", " +
                        "inferWeightLength: " + feature.dataLength()));
            }
        }
        Status tag;
        LOGGER.info(Common.addTag("[startFLJob] ----------------loading weight into inference " +
                "model-----------------"));
        status = Common.initSession(flParameter.getInferModelPath());
        if (status == FLClientStatus.FAILED) {
            retCode = ResponseCode.RequestError;
            return status;
        }
        tag = client.updateFeatures(flParameter.getInferModelPath(), inferFeatureMaps);
        Common.freeSession();
        if (!Status.SUCCESS.equals(tag)) {
            LOGGER.severe(Common.addTag("[startFLJob] unsolved error code in <Client.updateFeatures>"));
            retCode = ResponseCode.RequestError;
            return FLClientStatus.FAILED;
        }
        LOGGER.info(Common.addTag("[startFLJob] ----------------loading weight into train model-----------------"));
        status = Common.initSession(flParameter.getTrainModelPath());
        if (status == FLClientStatus.FAILED) {
            retCode = ResponseCode.RequestError;
            return status;
        }
        LOGGER.info(Common.addTag("[startFLJob] set <lr> for client: " + lr));
        tag = client.setLearningRate(lr);
        if (!Status.SUCCESS.equals(tag)) {
            LOGGER.severe(Common.addTag("[startFLJob] setLearningRate failed, return -1, please check"));
            retCode = ResponseCode.RequestError;
            return FLClientStatus.FAILED;
        }
        LOGGER.info(Common.addTag("[startFLJob] set <batch size> for client: " + batchSize));
        client.setBatchSize(batchSize);
        tag = client.updateFeatures(flParameter.getTrainModelPath(), trainFeatureMaps);
        Common.freeSession();
        if (!Status.SUCCESS.equals(tag)) {
            LOGGER.severe(Common.addTag("[startFLJob] unsolved error code in <Client.updateFeatures>"));
            retCode = ResponseCode.RequestError;
            return FLClientStatus.FAILED;
        }
        return status;
    }

    private FLClientStatus normalFeatures(ResponseFLJob flJob) {
        FLClientStatus status;
        Client client = ClientManager.getClient(flParameter.getFlName());
        int fmCount = flJob.featureMapLength();
        ArrayList<FeatureMap> featureMaps = new ArrayList<FeatureMap>();
        featureSize = 0;
        for (int i = 0; i < fmCount; i++) {
            FeatureMap feature = flJob.featureMap(i);
            if (feature == null) {
                LOGGER.severe(Common.addTag("[startFLJob] the feature returned from server is null"));
                retCode = ResponseCode.SystemError;
                return FLClientStatus.FAILED;
            }
            String featureName = feature.weightFullname();
            featureMaps.add(feature);
            featureSize += feature.dataLength();
            updateFeatureName.add(featureName);
            LOGGER.info(Common.addTag("[startFLJob] weightFullname: " + feature.weightFullname() + ", " +
                    "weightLength: " + feature.dataLength()));
        }
        Status tag;
        LOGGER.info(Common.addTag("[startFLJob] ----------------loading weight into model-----------------"));
        status = Common.initSession(flParameter.getTrainModelPath());
        if (status == FLClientStatus.FAILED) {
            retCode = ResponseCode.RequestError;
            return status;
        }
        LOGGER.info(Common.addTag("[startFLJob] set <lr> for client: " + lr));
        tag = client.setLearningRate(lr);
        if (!Status.SUCCESS.equals(tag)) {
            LOGGER.severe(Common.addTag("[startFLJob] setLearningRate failed, return -1, please check"));
            retCode = ResponseCode.RequestError;
            return FLClientStatus.FAILED;
        }
        LOGGER.info(Common.addTag("[startFLJob] set <batch size> for client: " + batchSize));
        client.setBatchSize(batchSize);
        tag = client.updateFeatures(flParameter.getTrainModelPath(), featureMaps);
        LOGGER.info(Common.addTag("[startFLJob] ===========free session============="));
        Common.freeSession();
        if (!Status.SUCCESS.equals(tag)) {
            LOGGER.severe(Common.addTag("[startFLJob] unsolved error code in <Client.updateFeatures>"));
            retCode = ResponseCode.RequestError;
            return FLClientStatus.FAILED;
        }
        return status;
    }

    private FLClientStatus parseResponseFeatures(ResponseFLJob flJob) {
        FLClientStatus status;
        int fmCount = flJob.featureMapLength();
        updateFeatureName.clear();

        if (localFLParameter.getServerMod().equals(ServerMod.HYBRID_TRAINING.toString())) {
            LOGGER.info(Common.addTag("[startFLJob] parseResponseFeatures by " + localFLParameter.getServerMod()));
            status = hybridFeatures(flJob);
            if (status == FLClientStatus.FAILED) {
                return status;
            }
        } else if (localFLParameter.getServerMod().equals(ServerMod.FEDERATED_LEARNING.toString())) {
            LOGGER.info(Common.addTag("[startFLJob] parseResponseFeatures by " + localFLParameter.getServerMod()));
            status = normalFeatures(flJob);
            if (status == FLClientStatus.FAILED) {
                return status;
            }
        }
        return FLClientStatus.SUCCESS;
    }

    private FLClientStatus deprecatedParseFeatures(ResponseFLJob flJob) {
        FLClientStatus status = FLClientStatus.SUCCESS;
        if (ALBERT.equals(flParameter.getFlName())) {
            LOGGER.info(Common.addTag("[startFLJob] into <parseResponseAlbert>"));
            status = deprecatedParseResponseAlbert(flJob);
        }
        if (LENET.equals(flParameter.getFlName())) {
            LOGGER.info(Common.addTag("[startFLJob] into <parseResponseLenet>"));
            status = deprecatedParseResponseLenet(flJob);
        }
        if (DEEPFM.equals(flParameter.getFlName())) {
            LOGGER.info(Common.addTag("[startFLJob] into <parseResponseDeepfm>"));
            status = deprecatedParseResponseDeepfm(flJob);
        }
        return status;
    }


    /**
     * response res
     *
     * @param flJob ResponseFLJob
     * @return FLClientStatus
     */
    public FLClientStatus doResponse(ResponseFLJob flJob) {
        if (flJob == null) {
            LOGGER.severe(Common.addTag("[startFLJob] the input parameter flJob is null"));
            retCode = ResponseCode.SystemError;
            return FLClientStatus.FAILED;
        }
        FLPlan flPlanConfig = flJob.flPlanConfig();
        if (flPlanConfig == null) {
            LOGGER.severe(Common.addTag("[startFLJob] the flPlanConfig is null"));
            retCode = ResponseCode.SystemError;
            return FLClientStatus.FAILED;
        }

        retCode = flJob.retcode();
        LOGGER.info(Common.addTag("[startFLJob] ==========the response message of startFLJob is:================"));
        LOGGER.info(Common.addTag("[startFLJob] return retCode: " + retCode));
        LOGGER.info(Common.addTag("[startFLJob] reason: " + flJob.reason()));
        LOGGER.info(Common.addTag("[startFLJob] iteration: " + flJob.iteration()));
        LOGGER.info(Common.addTag("[startFLJob] is selected: " + flJob.isSelected()));
        LOGGER.info(Common.addTag("[startFLJob] next request time: " + flJob.nextReqTime()));
        nextRequestTime = flJob.nextReqTime();
        LOGGER.info(Common.addTag("[startFLJob] timestamp: " + flJob.timestamp()));
        FLClientStatus status;
        int responseRetCode = flJob.retcode();

        switch (responseRetCode) {
            case (ResponseCode.SUCCEED):
                if (flJob.featureMapLength() <= 0) {
                    LOGGER.severe(Common.addTag("[startFLJob] the feature size get from server is zero"));
                    retCode = ResponseCode.SystemError;
                    return FLClientStatus.FAILED;
                }
                localFLParameter.setServerMod(flPlanConfig.serverMode());
                if (flPlanConfig.lr() != 0) {
                    lr = flPlanConfig.lr();
                } else {
                    LOGGER.info(Common.addTag("[startFLJob] the GlobalParameter <lr> from server: " + lr + " is not " +
                            "valid, " +
                            "will use the default value 0.1"));
                }
                batchSize = flPlanConfig.miniBatch();
                if (Common.checkFLName(flParameter.getFlName())) {
                    status = deprecatedParseFeatures(flJob);
                } else {
                    LOGGER.info(Common.addTag("[startFLJob] into <parseResponseFeatures>"));
                    status = parseResponseFeatures(flJob);
                }
                return status;
            case (ResponseCode.OutOfTime):
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.info(Common.addTag("[startFLJob] catch RequestError or SystemError"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[startFLJob] the return <retCode> from server is invalid: " + retCode));
                return FLClientStatus.FAILED;
        }
    }

    class RequestStartFLJobBuilder {
        private RequestFLJob requestFLJob;
        private FlatBufferBuilder builder;
        private int nameOffset = 0;
        private int iteration = 0;
        private int dataSize = 0;
        private int timestampOffset = 0;
        private int idOffset = 0;
        private int signDataOffset = 0;
        private int keyAttestationOffset = 0;
        private int equipCertOffset = 0;
        private int equipCACertOffset = 0;
        private int rootCertOffset = 0;

        public RequestStartFLJobBuilder() {
            builder = new FlatBufferBuilder();
        }

        /**
         * set flName
         *
         * @param name String
         * @return RequestStartFLJobBuilder
         */
        public RequestStartFLJobBuilder flName(String name) {
            if (name == null || name.isEmpty()) {
                LOGGER.severe(Common.addTag("[startFLJob] the parameter of <name> is null or empty, please check!"));
                throw new IllegalArgumentException();
            }
            this.nameOffset = this.builder.createString(name);
            return this;
        }

        /**
         * set id
         *
         * @param id String
         * @return RequestStartFLJobBuilder
         */
        public RequestStartFLJobBuilder id(String id) {
            if (id == null || id.isEmpty()) {
                LOGGER.severe(Common.addTag("[startFLJob] the parameter of <id> is null or empty, please check!"));
                throw new IllegalArgumentException();
            }
            this.idOffset = this.builder.createString(id);
            return this;
        }

        /**
         * set time
         *
         * @param timestamp long
         * @return RequestStartFLJobBuilder
         */
        public RequestStartFLJobBuilder time(long timestamp) {
            this.timestampOffset = builder.createString(String.valueOf(timestamp));
            return this;
        }

        /**
         * set dataSize
         *
         * @param dataSize int
         * @return RequestStartFLJobBuilder
         */
        public RequestStartFLJobBuilder dataSize(int dataSize) {
            // temp code need confirm
            this.dataSize = dataSize;
            LOGGER.info(Common.addTag("[startFLJob] the train data size: " + dataSize));
            return this;
        }

        /**
         * set iteration
         *
         * @param iteration iteration
         * @return RequestStartFLJobBuilder
         */
        public RequestStartFLJobBuilder iteration(int iteration) {
            this.iteration = iteration;
            return this;
        }

        /**
         * signData
         *
         * @param signData byte[]
         * @return RequestStartFLJobBuilder
         */
        public RequestStartFLJobBuilder signData(byte[] signData) {
            if (signData == null || signData.length == 0) {
                LOGGER.severe(
                        Common.addTag("[startFLJob] the parameter of <signData> is null or empty, please check!"));
                throw new IllegalArgumentException();
            }
            this.signDataOffset = RequestFLJob.createSignDataVector(builder, signData);
            return this;
        }

        /**
         * set certificateChain
         *
         * @param certificates Certificate array
         * @return RequestStartFLJobBuilder
         */
        public RequestStartFLJobBuilder certificateChain(Certificate[] certificates) {
            if (certificates == null || certificates.length < 4) {
                LOGGER.severe(Common.addTag("[startFLJob] the parameter of <certificates> is null or the length "
                        + "is not valid (should be >= 4), please check!"));
                throw new IllegalArgumentException();
            }
            try {
                String keyAttestationPem = PkiUtil.getPemFormat(certificates[0]);
                String equipCertPem = PkiUtil.getPemFormat(certificates[1]);
                String equipCACertPem = PkiUtil.getPemFormat(certificates[2]);
                String rootCertPem = PkiUtil.getPemFormat(certificates[3]);

                this.keyAttestationOffset = this.builder.createString(keyAttestationPem);
                this.equipCertOffset = this.builder.createString(equipCertPem);
                this.equipCACertOffset = this.builder.createString(equipCACertPem);
                this.rootCertOffset = this.builder.createString(rootCertPem);
            } catch (IOException e) {
                LOGGER.severe(Common.addTag("[StartFLJob] catch IOException in certificateChain: " + e.getMessage()));
            }
            return this;
        }

        /**
         * build protobuffer
         *
         * @return byte[] data
         */
        public byte[] build() {
            RequestFLJob.startRequestFLJob(this.builder);
            RequestFLJob.addFlName(builder, nameOffset);
            RequestFLJob.addFlId(builder, idOffset);
            RequestFLJob.addIteration(builder, iteration);
            RequestFLJob.addDataSize(builder, dataSize);
            RequestFLJob.addTimestamp(builder, timestampOffset);
            RequestFLJob.addSignData(builder, signDataOffset);
            RequestFLJob.addRootCert(builder, rootCertOffset);
            RequestFLJob.addEquipCaCert(builder, equipCACertOffset);
            RequestFLJob.addEquipCert(builder, equipCertOffset);
            RequestFLJob.addKeyAttestation(builder, keyAttestationOffset);
            int root = RequestFLJob.endRequestFLJob(builder);
            builder.finish(root);
            return builder.sizedByteArray();
        }
    }
}
