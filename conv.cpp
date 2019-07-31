void conv2d(const float *X, const float *W, float *Y,
            int batch,
            int fil_h, int fil_w,
            int in_h, int in_w, int in_ch,
            int ou_h, int ou_w, int ou_ch,
            int strides1, int strides2) {
    //puts("in conv2d");
    float *X_col = new float[ou_h * ou_w * fil_h * fil_w * in_ch];
    float *Y_batch = Y;
    const float *X_batch = X;
    int X_batch_dlt = in_h * in_w * in_ch;
    int Y_batch_dlt = ou_h * ou_w * ou_ch;
    int subX_col_dlt = fil_h * fil_w * in_ch;
    int xx_dlt = in_w * in_ch;
    //#pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        //printf("b %d\n", b);
        float *subX_col = X_col;
        for (int i = 0, p = 0; i < ou_h; ++i, p += strides1) {
            for (int j = 0, q = 0; j < ou_w; ++j, q += strides2) {
                int index = 0;
                for (int x = p, xx = p * xx_dlt; x < p + fil_h; ++x, xx += xx_dlt)
                    for (int y = q, yy = q * in_ch; y < q + fil_w; ++y, yy += in_ch)
                        for (int k = 0; k < in_ch; ++k)
                            subX_col[index++] = X_batch[xx + yy + k];
                subX_col += subX_col_dlt;
            }
        }
        matmul(X_col, W, Y_batch,
               false, false,
               ou_h * ou_w,
               fil_h * fil_w * in_ch,
               ou_ch);
       X_batch += X_batch_dlt;
       Y_batch += Y_batch_dlt;
    }
    delete [] X_col;
    //puts("out conv2d");
}

void conv2d_gradient_x(const float *D, const float *W, float *DX,
                       int batch,
                       int fil_h, int fil_w,
                       int in_h, int in_w, int in_ch,
                       int ou_h, int ou_w, int ou_ch,
                       int strides1, int strides2) {
    //puts("in conv2d_gradient_x");
    float *DX_col = new float[ou_h * ou_w * fil_h * fil_w * in_ch];
    float *DX_batch = DX;
    const float *D_batch = D;
    int D_batch_dlt = ou_h * ou_w * ou_ch;
    int DX_batch_dlt = in_h * in_w * in_ch;
    int subDX_col_dlt = fil_h * fil_w * in_ch;
    int xx_dlt = fil_w * in_ch;
    //#pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        //printf("b %d\n", b);
        float *subDX_col = DX_col;
        matmul(D_batch, W, DX_col,
               false, true,
               ou_h * ou_w,
               ou_ch,
               fil_h * fil_w * in_ch);
        for (int i = 0, p = 0; i < ou_h; ++i, p += strides1) {
            for (int j = 0, q = 0; j < ou_w; ++j, q += strides2) {
                int index = 0;
                for (int x = p, xx = p * xx_dlt; x < p + fil_h; ++x, xx += xx_dlt)
                    for (int y = q, yy = q * in_ch; y < q + fil_w; ++y, yy += in_ch)
                        for (int k = 0; k < in_ch; ++k)
                            DX_batch[xx + yy + k] = subDX_col[index++];
                subDX_col += subDX_col_dlt;
            }
        }
        D_batch += D_batch_dlt;
        DX_batch += DX_batch_dlt;
    }
    delete [] DX_col;
    //puts("out conv2d_gradient_x");
}

void conv2d_gradient_w(const float *D, const float *X, float *DW,
                       int batch,
                       int fil_h, int fil_w,
                       int in_h, int in_w, int in_ch,
                       int ou_h, int ou_w, int ou_ch,
                       int strides1, int strides2) {
    //puts("in conv2d_gradient_w");
    float *DW_batch = new float[fil_h * fil_w * in_ch * ou_ch];
    float *X_col = new float[ou_h * ou_w * fil_h * fil_w * in_ch];
    const float *D_batch = D;
    const float *X_batch = X;
    int D_batch_dlt = ou_h * ou_w * ou_ch;
    int X_batch_dlt = in_h * in_w * in_ch;
    int subDX_col_dlt = fil_h * fil_w * in_ch;
    int xx_dlt = in_w * in_ch;
    //#pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        float *subX_col = X_col;
        for (int i = 0, p = 0; i < ou_h; ++i, p += strides1) {
            for (int j = 0, q = 0; j < ou_w; ++j, q += strides2) {
                int index = 0;
                for (int x = p, xx = p * xx_dlt; x < p + fil_h; ++x, xx += xx_dlt)
                    for (int y = q, yy = q * in_ch; y < q + fil_w; ++y, yy += in_ch)
                        for (int k = 0; k < in_ch; ++k)
                            subX_col[index++] = X_batch[xx + yy + k];
                subX_col += subDX_col_dlt;
            }
        }
        matmul(X_col, D_batch, DW_batch,
               true, false,
               fil_h * fil_w * in_ch,
               ou_h * ou_w,
               ou_ch);
        for (int i = 0; i < fil_h * fil_w * in_ch * ou_ch; ++i) DW[i] += DW_batch[i];
        X_batch += X_batch_dlt;
        D_batch += D_batch_dlt;
    }
    delete [] DW_batch;
    delete [] X_col;
    //puts("out conv2d_gradient_w");
}
