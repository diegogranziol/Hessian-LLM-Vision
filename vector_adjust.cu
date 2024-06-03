// vector_adjust.cu
extern "C" __global__ void vector_adjust(const float* grad_vector, const float* V, const float* eigvals, float* adjusted_grad_vector, int num_eigenvalues, int vec_len, float delta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_len) {
        float adjustment = 0.0f;
        for (int i = 0; i < num_eigenvalues; i++) {
            float dot_product = 0.0f;
            for (int j = 0; j < vec_len; j++) {
                dot_product += grad_vector[j] * V[i * vec_len + j];
            }
            adjustment += (1.0f / eigvals[i] - 1.0f / (eigvals[i] + delta)) * dot_product * V[i * vec_len + idx];
        }
        adjusted_grad_vector[idx] += adjustment;
    }
}
