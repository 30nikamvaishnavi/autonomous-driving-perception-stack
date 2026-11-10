int32 id
string class_name
float32 confidence
float32 x1
float32 y1
float32 x2
float32 y2
float32 distance_m       # NaN until estimated by fusion_node
float32 lateral_m
float32 rel_speed_mps    # d(distance)/dt, negative = closing
float32 ttc_s            # +inf if not closing
