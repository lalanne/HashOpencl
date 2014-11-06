
__kernel //__attribute__ ((reqd_work_group_size(1, 1024, 1)))
void hash(__global char* key, 
        const unsigned int len,
        const unsigned int initval)
{
    printf("kernel.......len[%d]initval[%d]\n", len, initval);

    int r = get_global_id(0);
    int c = get_global_id(1);
    int gid0 = get_group_id(0);
    int gid1 = get_group_id(1);
    int rank = get_global_size(0);
    float running = 0.0f;
    int tmp = r*rank + c;
    printf("r[%d] c[%d] gid0[%d] gid1[%d] value[%f]\n", r, c, gid0, gid1);

    return;
}
