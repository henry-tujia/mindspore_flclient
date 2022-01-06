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

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;
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
<<<<<<< HEAD
=======
    private int retCode = ResponseCode.RequestError;
>>>>>>> d13157c5e1f28b842e3ea3aa1ba9490e971b4546

    protected String trainningTIme;
    private static double[] predictParameters;
    private static double modelParameter;
    private static double modelBasicTime;
    private static double modelIntercept;
    private static HashMap<String, Vector<Long>> timeDataPerEpoch  = new HashMap<String, Vector<Long>>();

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

<<<<<<< HEAD
=======
    public int getRetCode() {
        return retCode;
    }

>>>>>>> d13157c5e1f28b842e3ea3aa1ba9490e971b4546
    /**
     * TODO
     * Get the prediction of trainning time.
     *
     * @param batchSize      the batchSize of trainning option.
     *
     */
    public void predictTrainningTime(int batchSize, int epoches){
        // TODO all the set number needs to be test.
        if (flParameter.getFlName().equals(ALBERT)) {
            modelParameter = 12;
            modelBasicTime = 686 + System.currentTimeMillis() % 100;
            predictParameters = new double[] {0.00000000e+00, 2.13054940e-01, -1.88135869e-02, 9.80505051e-01,
                                         -6.67649652e-01, 2.71300855e-03, 2.49566224e-02,  7.95762709e-06,
                                         -3.90761689e-05, -2.14748431e-04};
            modelIntercept = 7.3465573;
        } else if (flParameter.getFlName().equals(LENET)) {
            modelParameter = 5;
            modelBasicTime = 319 + System.currentTimeMillis() % 100;
            predictParameters = new double[]{0.00000000e+00, 2.13054940e-01, -1.88135869e-02, 9.80505051e-01,
                                         -6.67649652e-01, 2.71300855e-03, 2.49566224e-02,  7.95762709e-06,
                                         -3.90761689e-05, -2.14748431e-04};
            modelIntercept = 7.3465573;
        }

        String Pid = ManagementFactory.getRuntimeMXBean().getName();
        if(timeDataPerEpoch.containsKey(Pid)){
            Vector<Long> timeDataPerEpoch_ = timeDataPerEpoch.get(Pid);
            long avg = 0;
            int timeDataPerEpochLength = timeDataPerEpoch_.size();
            if(timeDataPerEpochLength > 3) {
                for(int i = timeDataPerEpochLength - 1; i >= timeDataPerEpochLength - 3; i--){
                    avg += timeDataPerEpoch_.get(i);
                }
                modelBasicTime = avg /= 3;
            } else {
                for(int i = 0; i < timeDataPerEpochLength; i++){
                    avg += timeDataPerEpoch_.get(i);
                }
                modelBasicTime = avg /= timeDataPerEpochLength;
            }
        }

        double sum = 0;
        double a = modelParameter;
        int b = batchSize;
        double c = modelBasicTime;
        double[] x_data = new double[]{1, a, b, c, a * a, a * b, a * c, b * b, b * c, c * c};
        for(int i = 0; i < x_data.length; i++){
            sum += x_data[i] * predictParameters[i];
        }
        sum += modelIntercept;
        sum *= epoches;
        trainningTIme = sum+"";
    }

    /**
     * Record the running time of an epoch of PID
     *
     * @param PID  the PID of the process which running the model.
     * @param time the real running time of an epoch.
     */
    public void addTrainningTime(String Pid, long time){
        if(timeDataPerEpoch.containsKey(Pid)){
            timeDataPerEpoch.get(Pid).add(time);
        } else {
            Vector<Long> PidData = new Vector<Long>();
            PidData.add(time);
            timeDataPerEpoch.put(Pid, PidData);
        }
    }

    /**
     * Get a flatBuffer builder of RequestUploadTrainningTime.
     *
     * @param iteration      current iteration of federated learning task.
     * @param secureProtocol the object that defines encryption and decryption methods.
     * @param batchSize  the batchSize of trainning option.
     * @return the flatBuffer builder of RequestUploadTrainningTime in byte[] format.
     */
    public byte[] getRequestUploadTrainningTime(int iteration, SecureProtocol secureProtocol, int batchSize, int epochs) {
        this.predictTrainningTime(batchSize, epochs);
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
        LOGGER.info(Common.addTag("[uploadTrainningTime] ==========next request time: " + response.nextReqTime()));
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
}