
__kernel 
void hash(__global char* key, 
        const unsigned int len,
        const unsigned int initval,
        unsigned int* out)
{
    printf("kernel.......len[%d]initval[%d]\n", len, initval);

    unsigned int  hash, i;
    for(hash = i = 0; i < len; ++i){
        hash += key[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }

    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);

    printf("hash[%u]\n", hash);
    out[0] = hash;

    return;
}
