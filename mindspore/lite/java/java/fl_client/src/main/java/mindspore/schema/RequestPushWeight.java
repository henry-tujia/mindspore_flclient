// automatically generated by the FlatBuffers compiler, do not modify

package mindspore.schema;

import java.nio.*;
import java.lang.*;
import java.util.*;
import com.google.flatbuffers.*;

@SuppressWarnings("unused")
public final class RequestPushWeight extends Table {
  public static RequestPushWeight getRootAsRequestPushWeight(ByteBuffer _bb) { return getRootAsRequestPushWeight(_bb, new RequestPushWeight()); }
  public static RequestPushWeight getRootAsRequestPushWeight(ByteBuffer _bb, RequestPushWeight obj) { _bb.order(ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { bb_pos = _i; bb = _bb; vtable_start = bb_pos - bb.getInt(bb_pos); vtable_size = bb.getShort(vtable_start); }
  public RequestPushWeight __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public int iteration() { int o = __offset(4); return o != 0 ? bb.getInt(o + bb_pos) : 0; }
  public FeatureMap featureMap(int j) { return featureMap(new FeatureMap(), j); }
  public FeatureMap featureMap(FeatureMap obj, int j) { int o = __offset(6); return o != 0 ? obj.__assign(__indirect(__vector(o) + j * 4), bb) : null; }
  public int featureMapLength() { int o = __offset(6); return o != 0 ? __vector_len(o) : 0; }

  public static int createRequestPushWeight(FlatBufferBuilder builder,
      int iteration,
      int feature_mapOffset) {
    builder.startObject(2);
    RequestPushWeight.addFeatureMap(builder, feature_mapOffset);
    RequestPushWeight.addIteration(builder, iteration);
    return RequestPushWeight.endRequestPushWeight(builder);
  }

  public static void startRequestPushWeight(FlatBufferBuilder builder) { builder.startObject(2); }
  public static void addIteration(FlatBufferBuilder builder, int iteration) { builder.addInt(0, iteration, 0); }
  public static void addFeatureMap(FlatBufferBuilder builder, int featureMapOffset) { builder.addOffset(1, featureMapOffset, 0); }
  public static int createFeatureMapVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startFeatureMapVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static int endRequestPushWeight(FlatBufferBuilder builder) {
    int o = builder.endObject();
    return o;
  }
}

