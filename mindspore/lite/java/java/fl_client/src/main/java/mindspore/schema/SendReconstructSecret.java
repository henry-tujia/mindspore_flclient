// automatically generated by the FlatBuffers compiler, do not modify

package mindspore.schema;

import java.nio.*;
import java.lang.*;
import java.util.*;
import com.google.flatbuffers.*;

@SuppressWarnings("unused")
public final class SendReconstructSecret extends Table {
  public static SendReconstructSecret getRootAsSendReconstructSecret(ByteBuffer _bb) { return getRootAsSendReconstructSecret(_bb, new SendReconstructSecret()); }
  public static SendReconstructSecret getRootAsSendReconstructSecret(ByteBuffer _bb, SendReconstructSecret obj) { _bb.order(ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { bb_pos = _i; bb = _bb; vtable_start = bb_pos - bb.getInt(bb_pos); vtable_size = bb.getShort(vtable_start); }
  public SendReconstructSecret __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public String flId() { int o = __offset(4); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer flIdAsByteBuffer() { return __vector_as_bytebuffer(4, 1); }
  public ByteBuffer flIdInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 4, 1); }
  public ClientShare reconstructSecretShares(int j) { return reconstructSecretShares(new ClientShare(), j); }
  public ClientShare reconstructSecretShares(ClientShare obj, int j) { int o = __offset(6); return o != 0 ? obj.__assign(__indirect(__vector(o) + j * 4), bb) : null; }
  public int reconstructSecretSharesLength() { int o = __offset(6); return o != 0 ? __vector_len(o) : 0; }
  public int iteration() { int o = __offset(8); return o != 0 ? bb.getInt(o + bb_pos) : 0; }
  public String timestamp() { int o = __offset(10); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer timestampAsByteBuffer() { return __vector_as_bytebuffer(10, 1); }
  public ByteBuffer timestampInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 10, 1); }

  public static int createSendReconstructSecret(FlatBufferBuilder builder,
      int fl_idOffset,
      int reconstruct_secret_sharesOffset,
      int iteration,
      int timestampOffset) {
    builder.startObject(4);
    SendReconstructSecret.addTimestamp(builder, timestampOffset);
    SendReconstructSecret.addIteration(builder, iteration);
    SendReconstructSecret.addReconstructSecretShares(builder, reconstruct_secret_sharesOffset);
    SendReconstructSecret.addFlId(builder, fl_idOffset);
    return SendReconstructSecret.endSendReconstructSecret(builder);
  }

  public static void startSendReconstructSecret(FlatBufferBuilder builder) { builder.startObject(4); }
  public static void addFlId(FlatBufferBuilder builder, int flIdOffset) { builder.addOffset(0, flIdOffset, 0); }
  public static void addReconstructSecretShares(FlatBufferBuilder builder, int reconstructSecretSharesOffset) { builder.addOffset(1, reconstructSecretSharesOffset, 0); }
  public static int createReconstructSecretSharesVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startReconstructSecretSharesVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addIteration(FlatBufferBuilder builder, int iteration) { builder.addInt(2, iteration, 0); }
  public static void addTimestamp(FlatBufferBuilder builder, int timestampOffset) { builder.addOffset(3, timestampOffset, 0); }
  public static int endSendReconstructSecret(FlatBufferBuilder builder) {
    int o = builder.endObject();
    return o;
  }
}

