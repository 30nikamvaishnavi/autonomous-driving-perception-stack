std_msgs/Header header   # stamp = source image stamp (for end-to-end latency)
uint8 SAFE=0
uint8 CAUTION=1
uint8 WARNING=2
uint8 CRITICAL=3
uint8 level
int32 track_id           # -1 if none
float32 distance_m
float32 ttc_s
