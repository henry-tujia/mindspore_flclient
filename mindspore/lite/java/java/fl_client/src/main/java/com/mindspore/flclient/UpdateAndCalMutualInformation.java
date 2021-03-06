package com.mindspore.flclient;

import java.util.Date;
import java.util.Map;
import java.util.logging.Logger;

import com.google.flatbuffers.FlatBufferBuilder;
import com.mindspore.flclient.JavaMI.MutualInformation;

import mindspore.schema.RequestUpdateAndCalMutualInformation;
import mindspore.schema.ResponseCode;
import mindspore.schema.ResponseUpdateAndCalMutualInformation;

public class UpdateAndCalMutualInformation {
    private static final Logger LOGGER = Logger.getLogger(UpdateAndCalMutualInformation.class.toString());
    private static volatile UpdateAndCalMutualInformation updateAndCalMutualInformation;

    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private LocalFLParameter localFLParameter = LocalFLParameter.getInstance();
    private FLClientStatus status;

    private UpdateAndCalMutualInformation() {
    }

    /**
     * Get the singleton object of the class updateAndCalMutualInformation.
     *
     * @return the singleton object of the class updateAndCalMutualInformation.
     */
    public static UpdateAndCalMutualInformation getInstance() {
        UpdateAndCalMutualInformation localRef = updateAndCalMutualInformation;
        if (localRef == null) {
            synchronized (UpdateAndCalMutualInformation.class) {
                localRef = updateAndCalMutualInformation;
                if (localRef == null) {
                    updateAndCalMutualInformation = localRef = new UpdateAndCalMutualInformation();
                }
            }
        }
        return localRef;
    }

    public FLClientStatus getStatus() {
        return status;
    }

    

    /**
     * @description
     * @author ICT_tanhao
     * @date 2021/10/14
     **/
    private double calMutualInformation(Map<String, float[]> localModel,Map<String, float[]> serverModel) {
        
        double res = MutualInformation.calculateMutualInformation(MapToArray(localModel), MapToArray(serverModel));
        
        return res;

    }


    private double[] MapToArray(Map<String, float[]> input){
        int enuNum = 0;
        for (float[] b : input.values()) {
            enuNum+= b.length;
        }

        double[] res = new double[enuNum];

        int j = 0;

        for (float[] b : input.values()) {
            for(float c : b){
                res[j] = c;
                j += 1;
            }
        }

        return res;

    }

    /**
     * Get a flatBuffer builder of RequestUpdateAndCalMutualInformation.
     *
     * @param iteration      current iteration of federated learning task.
     * @param secureProtocol the object that defines encryption and decryption methods.
     * @param batchSize  the batchSize of trainning option.
     * @return the flatBuffer builder of RequestUpdateAndCalMutualInformation in byte[] format.
     */
    public byte[] getRequestUpdateAndCalMutualInformation(Map<String, float[]> localModel,Map<String, float[]> serverModel, int iteration) {
        RequestUpdateAndCalMutualInformationBuilder builder = new RequestUpdateAndCalMutualInformationBuilder();
        LOGGER.info(Common.addTag("[UpdateAndCalMutualInformation] ID is "+localFLParameter.getFlID()+", mul is "+calMutualInformation(localModel,serverModel)+", Iteration is "+ iteration));
        return builder.time().id(localFLParameter.getFlID()).mul(calMutualInformation(localModel,serverModel)+"").iteration(iteration).build();
    }

    /**
     * Handle the response message returned from server.
     *
     * @param response the response message returned from server.
     * @return the status code corresponding to the response message.
     */
    public FLClientStatus doResponse(ResponseUpdateAndCalMutualInformation response) {
        LOGGER.info(Common.addTag("[UpdateAndCalMutualInformation] ==========UpdateAndCalMutualInformation response================"));
        LOGGER.info(Common.addTag("[UpdateAndCalMutualInformation] ==========retcode: " + response.retcode()));
        LOGGER.info(Common.addTag("[UpdateAndCalMutualInformation] ==========reason: " + response.reason()));
        LOGGER.info(Common.addTag("[UpdateAndCalMutualInformation] ==========next request time: " + response.nextReqTime()));
        FLClientStatus status = FLClientStatus.SUCCESS;
        switch (response.retcode()) {
            case (ResponseCode.SUCCEED):
                LOGGER.info(Common.addTag("[UpdateAndCalMutualInformation] UpdateAndCalMutualInformation response success"));
                return status;
            case (ResponseCode.SucNotReady):
                LOGGER.info(Common.addTag("[UpdateAndCalMutualInformation] server is not ready now: need wait and request UpdateAndCalMutualInformation again"));
                return FLClientStatus.WAIT;
            case (ResponseCode.OutOfTime):
                LOGGER.info(Common.addTag("[UpdateAndCalMutualInformation] out of time: need wait and request startFLJob again"));
                return FLClientStatus.RESTART;
            case (ResponseCode.RequestError):
            case (ResponseCode.SystemError):
                LOGGER.warning(Common.addTag("[UpdateAndCalMutualInformation] catch RequestError or SystemError"));
                return FLClientStatus.FAILED;
            default:
                LOGGER.severe(Common.addTag("[UpdateAndCalMutualInformation] the return <retCode> from server is invalid: " + response.retcode()));
                return FLClientStatus.FAILED;
        }
    }

    class RequestUpdateAndCalMutualInformationBuilder {
        private RequestUpdateAndCalMutualInformation requestUC;
        private FlatBufferBuilder builder;
        private int idOffset = 0;
        private int mulOffset = 0;
        private int iteration = 0;
        private int timestampOffset = 0;

        private RequestUpdateAndCalMutualInformationBuilder() {
            builder = new FlatBufferBuilder();
        }

        /**
         * Serialize the element flName in RequestUpdateAndCalMutualInformation.
         *
         * @param name the model name.
         * @return the RequestUpdateAndCalMutualInformationBuilder object.
         */
        private RequestUpdateAndCalMutualInformationBuilder mul(String mul) {
            if (mul == null || mul.isEmpty()) {
                LOGGER.severe(Common.addTag("[uploadTrainningTime] the parameter of <mul> is null or empty, please check!"));
                throw new IllegalArgumentException();
            }
            this.mulOffset = this.builder.createString(mul);
            return this;
        }

        /**
         * Serialize the element timestamp in RequestUpdateAndCalMutualInformation.
         *
         * @return the RequestUpdateAndCalMutualInformationBuilder object.
         */
        private RequestUpdateAndCalMutualInformationBuilder time() {
            Date date = new Date();
            long time = date.getTime();
            this.timestampOffset = builder.createString(String.valueOf(time));
            return this;
        }

        /**
         * Serialize the element fl_id in RequestUpdateAndCalMutualInformation.
         *
         * @param id a number that uniquely identifies a client.
         * @return the RequestUpdateAndCalMutualInformationBuilder object.
         */
        private RequestUpdateAndCalMutualInformationBuilder id(String id) {
            if (id == null || id.isEmpty()) {
                LOGGER.severe(Common.addTag("[uploadTrainningTime] the parameter of <id> is null or empty, please check!"));
                throw new IllegalArgumentException();
            }
            this.idOffset = this.builder.createString(id);
            return this;
        }

        private RequestUpdateAndCalMutualInformationBuilder iteration(int iteration) {
            this.iteration = iteration;
            return this;
        }

        /**
         * Create a flatBuffer builder of RequestUpdateAndCalMutualInformation.
         *
         * @return the flatBuffer builder of RequestUpdateAndCalMutualInformation in byte[] format.
         */
        private byte[] build() {
            RequestUpdateAndCalMutualInformation.startRequestUpdateAndCalMutualInformation(builder);
            RequestUpdateAndCalMutualInformation.addFlId(builder, idOffset);
            RequestUpdateAndCalMutualInformation.addTimestamp(builder, timestampOffset);
            RequestUpdateAndCalMutualInformation.addIteration(builder, iteration);
            int root = RequestUpdateAndCalMutualInformation.endRequestUpdateAndCalMutualInformation(builder);
            builder.finish(root);
            return builder.sizedByteArray();
        }
    }
}
