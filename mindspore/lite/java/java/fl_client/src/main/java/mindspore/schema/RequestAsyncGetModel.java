// automatically generated by the FlatBuffers compiler, do not modify

package mindspore.schema;

import java.nio.*;
import java.lang.*;
import java.util.*;
import com.google.flatbuffers.*;

@SuppressWarnings("unused")
public final class RequestAsyncGetModel extends Table {
  public static RequestAsyncGetModel getRootAsRequestAsyncGetModel(ByteBuffer _bb) { return getRootAsRequestAsyncGetModel(_bb, new RequestAsyncGetModel()); }
  public static RequestAsyncGetModel getRootAsRequestAsyncGetModel(ByteBuffer _bb, RequestAsyncGetModel obj) { _bb.order(ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { bb_pos = _i; bb = _bb; vtable_start = bb_pos - bb.getInt(bb_pos); vtable_size = bb.getShort(vtable_start); }
  public RequestAsyncGetModel __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public String flName() { int o = __offset(4); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer flNameAsByteBuffer() { return __vector_as_bytebuffer(4, 1); }
  public ByteBuffer flNameInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 4, 1); }
  public int iteration() { int o = __offset(6); return o != 0 ? bb.getInt(o + bb_pos) : 0; }

  public static int createRequestAsyncGetModel(FlatBufferBuilder builder,
      int fl_nameOffset,
      int iteration) {
    builder.startObject(2);
    RequestAsyncGetModel.addIteration(builder, iteration);
    RequestAsyncGetModel.addFlName(builder, fl_nameOffset);
    return RequestAsyncGetModel.endRequestAsyncGetModel(builder);
  }

  public static void startRequestAsyncGetModel(FlatBufferBuilder builder) { builder.startObject(2); }
  public static void addFlName(FlatBufferBuilder builder, int flNameOffset) { builder.addOffset(0, flNameOffset, 0); }
  public static void addIteration(FlatBufferBuilder builder, int iteration) { builder.addInt(1, iteration, 0); }
  public static int endRequestAsyncGetModel(FlatBufferBuilder builder) {
    int o = builder.endObject();
    return o;
  }
}

