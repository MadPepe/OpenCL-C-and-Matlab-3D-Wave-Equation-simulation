__kernel void ProjectCUDA(__global const float *A, __global const float *B, __global float *C) {
    
    // Get the index of the current element to be processed
    int z = get_global_id(0);
    int y = get_global_id(1);
    int x = get_global_id(2);
    /*
    int x = get_local_id(0);
    int y = get_local_id(1);
    int z = get_local_id(2);*/
    
    /*int x = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int y = get_group_id(1) * get_local_size(1) + get_local_id(1);
    int z = get_group_id(2) * get_local_size(2) + get_local_id(2);
    */
    int i = x + y * 20 + z * 20 * 20;
    
    int previous_i = (x + y * 20 + z * 20 * 20)-400;
    int next_i = (x + y * 20 + z * 20 * 20)+400;
    
    int previous_j = (x + y * 20 + z * 20 * 20)-20;
    int next_j = (x + y * 20 + z * 20 * 20)+20;
    
    int previous_k = (x + y * 20 + z * 20 * 20)-1;
    int next_k = (x + y * 20 + z * 20 * 20)+1;
    
    
    // Do the operation
    // u[s+1][i][j][k] = (2*u[s][i][j][k] - u[s-1][i][j][k] + (Cx*Cx)*(u[s][i+1][j][k] + u[s][i-1][j][k] + u[s][i][j+1][k] + u[s][i][j-1][k] + u[s][i][j][k+1] + u[s][i][j][k-1] - 6*u[s][i][j][k]));
    
    if(x>0 && y>0 && z>0 && x<19 && y<19 && z<19) C[i] = (((2*B[i]) - A[i]) + (0.25*0.25)*(B[next_i] + B[previous_i] + B[next_j] + B[previous_j] + B[next_k] + B[previous_k] - (6*B[i])));
    //C[i] = z;
}
