// automatically generated by the FlatBuffers compiler, do not modify

package mindspore.schema;

import java.nio.*;
import java.lang.*;
import java.util.*;
import com.google.flatbuffers.*;

@SuppressWarnings("unused")
public final class ResponsePushWeight extends Table {
  public static ResponsePushWeight getRootAsResponsePushWeight(ByteBuffer _bb) { return getRootAsResponsePushWeight(_bb, new ResponsePushWeight()); }
  public static ResponsePushWeight getRootAsResponsePushWeight(ByteBuffer _bb, ResponsePushWeight obj) { _bb.order(ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { bb_pos = _i; bb = _bb; vtable_start = bb_pos - bb.getInt(bb_pos); vtable_size = bb.getShort(vtable_start); }
  public ResponsePushWeight __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public int retcode() { int o = __offset(4); return o != 0 ? bb.getInt(o + bb_pos) : 0; }
  public String reason() { int o = __offset(6); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer reasonAsByteBuffer() { return __vector_as_bytebuffer(6, 1); }
  public ByteBuffer reasonInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 6, 1); }
  public int iteration() { int o = __offset(8); return o != 0 ? bb.getInt(o + bb_pos) : 0; }

  public static int createResponsePushWeight(FlatBufferBuilder builder,
      int retcode,
      int reasonOffset,
      int iteration) {
    builder.startObject(3);
    ResponsePushWeight.addIteration(builder, iteration);
    ResponsePushWeight.addReason(builder, reasonOffset);
    ResponsePushWeight.addRetcode(builder, retcode);
    return ResponsePushWeight.endResponsePushWeight(builder);
  }

  public static void startResponsePushWeight(FlatBufferBuilder builder) { builder.startObject(3); }
  public static void addRetcode(FlatBufferBuilder builder, int retcode) { builder.addInt(0, retcode, 0); }
  public static void addReason(FlatBufferBuilder builder, int reasonOffset) { builder.addOffset(1, reasonOffset, 0); }
  public static void addIteration(FlatBufferBuilder builder, int iteration) { builder.addInt(2, iteration, 0); }
  public static int endResponsePushWeight(FlatBufferBuilder builder) {
    int o = builder.endObject();
    return o;
  }
}

