package com.mindspore.flclient;

import static com.mindspore.flclient.LocalFLParameter.ALBERT;
import static com.mindspore.flclient.LocalFLParameter.LENET;

import com.google.flatbuffers.FlatBufferBuilder;

import com.mindspore.flclient.model.AlTrainBert;
import com.mindspore.flclient.model.SessionUtil;
import com.mindspore.flclient.model.TrainLenet;

import mindspore.schema.RequestUploadTrainningTime;
import mindspore.schema.ResponseCode;
import mindspore.schema.ResponseUploadTrainningTime;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * @author ICT_hetianliu
 * 
 * Define the prediction of trainning time method, handle the response message returned from server for uploadTrainningTime request.
 *
 * @since 2021-10-15
 */

 public class UploadTrainningTime {
    private static final Logger LOGGER = Logger.getLogger(UploadTrainningTime.class.toString());
    private static volatile UploadTrainningTime uploadTrainningTime;

    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private FLParameter flParameter = FLParameter.getInstance();
    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private FLClientStatus status;

    protected String trainningTIme;

    private UploadTrainningTime() {
    }

    /**
     * Get the singleton object of the class UploadTrainningTime.
     *
     * @return the singleton object of the class UploadTrainningTime.
     */
    public static UploadTrainningTime getInstance() {
        UploadTrainningTime localRef = uploadTrainningTime;
        if (localRef == null) {
            synchronized (UploadTrainningTime.class) {
                localRef = uploadTrainningTime;
                if (localRef == null) {
                    uploadTrainningTime = localRef = new UploadTrainningTime();
                }
            }
        }
        return localRef;
    }

    public FLClientStatus getStatus() {
        return status;
    }

    /**
     * TODO
     * Get the prediction of trainning time.
     *
     * @param batchSize      the batchSize of trainning option.
     *
     */
    public void predictTrainningTime(int batchSize){

    }

    /**
     * Get a flatBuffer builder of RequestUploadTrainningTime.
     *
     * @param iteration      current iteration of federated learning task.
     * @param secureProtocol the object that defines encryption and decryption methods.
     * @param batchSize  the batchSize of trainning option.
     * @return the flatBuffer builder of RequestUploadTrainningTime in byte[] format.
     */
    public byte[] getRequestUploadTrainningTime(int iteration, SecureProtocol secureProtocol, int batchSize) {
        this.predictTrainningTime(batchSize);
        RequestUploadTrainningTimeBuilder builder = new RequestUploadTrainningTimeBuilder(localFLParameter.getEncryptLevel());
        return builder.flName(flParameter.getFlName()).time().id(localFLParameter.getFlID())
                .trainningTime(this.trainningTIme).iteration(iteration).build();
    }

    /**
     * Handle the response message returned from server.
     *
     * @param response the response message returned from server.
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus doResponse(ResponseUploadTrainningTime response) {
        LOGGER.info(Common.addTag("[uploadTrainningTime] ==========uploadTrainningTime response================"));
        LOGGER.info(Common.addTag("[uploadTrainningTime] ==========retcode: " + response.retcode()));
        LOGGER.info(Common.addTag("[uploadTrainningTime] ==========reason: " + response.reason()));
        switch (response.retcode()) {
            case (ResponseCode.SUCCEED):
                LOGGER.info(Common.addTag("[uploadTrainningTime] uploadTrainningTime success"));
                return FLClientStatus.SUCCESS;
            case (ResponseCode.OutOfTime):
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.warning(Common.addTag("[uploadTrainningTime] catch RequestError or SystemError"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[uploadTrainningTime]the return <retCode> from server is invalid: " +
                        response.retcode()));
                return FLClientStatus.FAILED;
        }
    }

    class RequestUploadTrainningTimeBuilder {
        private RequestUploadTrainningTime requestUM;
        private FlatBufferBuilder builder;
        private int nameOffset = 0;
        private int idOffset = 0;
        private int timestampOffset = 0;
        private int iteration = 0;
        private int trainningTimeOffset = 0;
        private EncryptLevel encryptLevel = EncryptLevel.NOT_ENCRYPT;

        private RequestUploadTrainningTimeBuilder(EncryptLevel encryptLevel) {
            builder = new FlatBufferBuilder();
            this.encryptLevel = encryptLevel;
        }

        /**
         * Serialize the element flName in RequestUploadTrainningTime.
         *
         * @param name the model name.
         * @return the RequestUploadTrainningTimeBuilder object.
         */
        private RequestUploadTrainningTimeBuilder flName(String name) {
            if (name == null || name.isEmpty()) {
                LOGGER.severe(Common.addTag("[uploadTrainningTime] the parameter of <name> is null or empty, please check!"));
                throw new IllegalArgumentException();
            }
            this.nameOffset = this.builder.createString(name);
            return this;
        }

        /**
         * Serialize the element timestamp in RequestUploadTrainningTime.
         *
         * @return the RequestUploadTrainningTimeBuilder object.
         */
        private RequestUploadTrainningTimeBuilder time() {
            Date date = new Date();
            long time = date.getTime();
            this.timestampOffset = builder.createString(String.valueOf(time));
            return this;
        }

        /**
         * Serialize the element iteration in RequestUploadTrainningTime.
         *
         * @param iteration current iteration of federated learning task.
         * @return the RequestUploadTrainningTimeBuilder object.
         */
        private RequestUploadTrainningTimeBuilder iteration(int iteration) {
            this.iteration = iteration;
            return this;
        }

        /**
         * Serialize the element fl_id in RequestUploadTrainningTime.
         *
         * @param id a number that uniquely identifies a client.
         * @return the RequestUploadTrainningTimeBuilder object.
         */
        private RequestUploadTrainningTimeBuilder id(String id) {
            if (id == null || id.isEmpty()) {
                LOGGER.severe(Common.addTag("[uploadTrainningTime] the parameter of <id> is null or empty, please check!"));
                throw new IllegalArgumentException();
            }
            this.idOffset = this.builder.createString(id);
            return this;
        }

        /**
         * Serialize the element trainning_time in RequestUploadTrainningTime.
         *
         * @param trainningTime A predicted trainning Time of Model Running.
         * @return the RequestUploadTrainningTimeBuilder object.
         */
        private RequestUploadTrainningTimeBuilder trainningTime(String trainningTime) {
             if (trainningTime == null || trainningTime.isEmpty()) {
                LOGGER.severe(Common.addTag("[uploadTrainningTime] the parameter of <trainningTime> is null or empty, please check!"));
                throw new IllegalArgumentException();
            }
            this.trainningTimeOffset = this.builder.createString(trainningTime);
            return this;
        }


        /**
         * Create a flatBuffer builder of RequestUploadTrainningTime.
         *
         * @return the flatBuffer builder of RequestUploadTrainningTime in byte[] format.
         */
        private byte[] build() {
            RequestUploadTrainningTime.startRequestUploadTrainningTime(this.builder);
            RequestUploadTrainningTime.addFlName(builder, nameOffset);
            RequestUploadTrainningTime.addFlId(this.builder, idOffset);
            RequestUploadTrainningTime.addTimestamp(builder, this.timestampOffset);
            RequestUploadTrainningTime.addIteration(builder, this.iteration);
            RequestUploadTrainningTime.addTrainningTime(builder, this.trainningTimeOffset);
            int root = RequestUploadTrainningTime.endRequestUploadTrainningTime(builder);
            builder.finish(root);
            return builder.sizedByteArray();
        }
    }