
__kernel //__attribute__ ((reqd_work_group_size(1, 1024, 1)))
void naive(__global float* a, __global float* b, __global float* output)
{
  int r = get_global_id(0);
  int c = get_global_id(1);
  int gid0 = get_group_id(0);
  int gid1 = get_group_id(1);
  int rank = get_global_size(0);
  float running = 0.0f;
  int tmp = r*rank + c;
  int value = b[r*rank + c];
  printf("r[%d] c[%d] gid0[%d] gid1[%d] value[%f]\n", r, c, gid0, gid1, value);

  return;
}
