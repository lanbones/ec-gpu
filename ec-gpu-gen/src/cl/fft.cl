/*
 * FFT algorithm is inspired from: http://www.bealto.com/gpu-fft_group-1.html
 */
KERNEL void FIELD_radix_fft(const GLOBAL FIELD* x, // Source buffer
                      GLOBAL FIELD* y, // Destination buffer
                      const GLOBAL FIELD* pq, // Precalculated twiddle factors
                      GLOBAL FIELD* omegas, // [omega, omega^2, omega^4, ...]
                      LOCAL FIELD* u_arg, // Local buffer to store intermediary values
                      uint n, // Number of elements
                      uint lgp, // Log2 of `p` (Read more in the link above)
                      uint deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                      uint max_deg) // Maximum degree supported, according to `pq` and `omegas`
{
// CUDA doesn't support local buffers ("shared memory" in CUDA lingo) as function arguments,
// ignore that argument and use the globally defined extern memory instead.
#ifdef CUDA
  // There can only be a single dynamic shared memory item, hence cast it to the type we need.
  FIELD* u = (FIELD*)cuda_shared;
#else
  LOCAL FIELD* u = u_arg;
#endif

  uint lid = GET_LOCAL_ID();
  uint lsize = GET_LOCAL_SIZE();
  uint index = GET_GROUP_ID();
  uint t = n >> deg;
  uint p = 1 << lgp;
  uint k = index & (p - 1);

  x += index;
  y += ((index - k) << deg) + k;

  uint count = 1 << deg; // 2^deg
  uint counth = count >> 1; // Half of count

  uint counts = count / lsize * lid;
  uint counte = counts + count / lsize;

  // Compute powers of twiddle
  const FIELD twiddle = FIELD_pow_lookup(omegas, (n >> lgp >> deg) * k);
  FIELD tmp = FIELD_pow(twiddle, counts);
  for(uint i = counts; i < counte; i++) {
    u[i] = FIELD_mul(tmp, x[i*t]);
    tmp = FIELD_mul(tmp, twiddle);
  }
  BARRIER_LOCAL();

  const uint pqshift = max_deg - deg;
  for(uint rnd = 0; rnd < deg; rnd++) {
    const uint bit = counth >> rnd;
    for(uint i = counts >> 1; i < counte >> 1; i++) {
      const uint di = i & (bit - 1);
      const uint i0 = (i << 1) - di;
      const uint i1 = i0 + bit;
      tmp = u[i0];
      u[i0] = FIELD_add(u[i0], u[i1]);
      u[i1] = FIELD_sub(tmp, u[i1]);
      if(di != 0) u[i1] = FIELD_mul(pq[di << rnd << pqshift], u[i1]);
    }

    BARRIER_LOCAL();
  }

  for(uint i = counts >> 1; i < counte >> 1; i++) {
    y[i*p] = u[bitreverse(i, deg)];
    y[(i+counth)*p] = u[bitreverse(i + counth, deg)];
  }
}

KERNEL void FIELD_distribute_powers_zeta(
  GLOBAL FIELD* value,
  GLOBAL FIELD* coset_powers,
  uint coset_powers_n
) {
  const uint gid = GET_GLOBAL_ID();
  uint index = gid % coset_powers_n;
  if (index != 0) {
    value[gid] = FIELD_mul(value[gid], coset_powers[index - 1]);
  }
}

/// Multiplies all of the elements by `field`
KERNEL void FIELD_mul_by_field(GLOBAL FIELD* elements,
                        uint n,
                        FIELD field) {
  const uint gid = GET_GLOBAL_ID();
  elements[gid] = FIELD_mul(elements[gid], field);
}

KERNEL void FIELD_eval_h_permutation_part1(
  GLOBAL FIELD* value,
  GLOBAL FIELD* first_set,
  GLOBAL FIELD* last_set,
  GLOBAL FIELD* l0,
  GLOBAL FIELD* l_last,
  GLOBAL FIELD* l_active_row,
  GLOBAL FIELD* y_beta_gamma
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;

  // l_0(X) * (1 - z_0(X)) = 0
  value[idx] = FIELD_mul(value[idx], y_beta_gamma[0]);
  FIELD tmp = FIELD_sub(FIELD_ONE, first_set[idx]);
  tmp = FIELD_mul(tmp, l0[idx]);
  value[idx] = FIELD_add(value[idx], tmp);

  // l_last(X) * (z_l(X)^2 - z_l(X)) = 0
  value[idx] = FIELD_mul(value[idx], y_beta_gamma[0]);
  tmp = FIELD_mul(last_set[idx], last_set[idx]);
  tmp = FIELD_sub(tmp, last_set[idx]);
  tmp = FIELD_mul(tmp, l_last[idx]);
  value[idx] = FIELD_add(value[idx], tmp);
}

KERNEL void FIELD_eval_h_permutation_part2(
  GLOBAL FIELD* value,
  GLOBAL FIELD* curr_set,
  GLOBAL FIELD* prev_set,
  GLOBAL FIELD* l0,
  GLOBAL FIELD* y_beta_gamma,
  uint rot,
  uint size
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;

  uint r_prev = (idx + size + rot) & (size - 1);

  // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
  value[idx] = FIELD_mul(value[idx], y_beta_gamma[0]);
  FIELD tmp = FIELD_sub(curr_set[idx], prev_set[r_prev]);
  tmp = FIELD_mul(tmp, l0[idx]);
  value[idx] = FIELD_add(value[idx], tmp);
}

KERNEL void FIELD_eval_h_permutation_left_prepare(
  GLOBAL FIELD* left,
  GLOBAL FIELD* permutation,
  uint rot,
  uint size
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;

  uint r_prev = (idx + size + rot) & (size - 1);
  left[idx] = permutation[r_prev];
}


KERNEL void FIELD_eval_h_permutation_left_right(
  GLOBAL FIELD* left,
  GLOBAL FIELD* right,
  GLOBAL FIELD* origin,
  GLOBAL FIELD* permutation,
  GLOBAL FIELD* current_delta,
  GLOBAL FIELD* y_beta_gamma_delta
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;

  FIELD tmp = FIELD_mul(y_beta_gamma_delta[1], permutation[idx]);
  tmp = FIELD_add(tmp, y_beta_gamma_delta[2]);
  tmp = FIELD_add(tmp, origin[idx]);
  left[idx] = FIELD_mul(left[idx], tmp);

  tmp = FIELD_add(current_delta[idx], y_beta_gamma_delta[2]);
  tmp = FIELD_add(tmp, origin[idx]);
  right[idx] = FIELD_mul(right[idx], tmp);

  current_delta[idx] = FIELD_mul(current_delta[idx], y_beta_gamma_delta[3]);
}

KERNEL void FIELD_eval_h_permutation_part3(
  GLOBAL FIELD* value,
  GLOBAL FIELD* left,
  GLOBAL FIELD* right,
  GLOBAL FIELD* l_active_row,
  GLOBAL FIELD* y_beta_gamma
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;

  value[idx] = FIELD_mul(value[idx], y_beta_gamma[0]);
  FIELD tmp = FIELD_sub(left[idx], right[idx]);
  tmp = FIELD_mul(tmp, l_active_row[idx]);
  value[idx] = FIELD_add(value[idx], tmp);
}

KERNEL void FIELD_eval_h_lookups(
  GLOBAL FIELD* values,
  GLOBAL FIELD* table,
  GLOBAL FIELD* permuted_input_coset,
  GLOBAL FIELD* permuted_table_coset,
  GLOBAL FIELD* product_coset,
  GLOBAL FIELD* l0,
  GLOBAL FIELD* l_last,
  GLOBAL FIELD* l_active_row,
  GLOBAL FIELD* y_beta_gamma,
  uint rot,
  uint size
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;

  uint r_next = (idx + rot) & (size - 1);
  uint r_prev = (idx + size - rot) & (size - 1);

  FIELD value = values[idx];

  // l_0(X) * (1 - z(X)) = 0
  value = FIELD_mul(value, y_beta_gamma[0]);
  FIELD tmp = FIELD_sub(FIELD_ONE, product_coset[idx]);
  tmp = FIELD_mul(tmp, l0[idx]);
  value = FIELD_add(value, tmp);

  // l_last(X) * (z(X)^2 - z(X)) = 0
  value = FIELD_mul(value, y_beta_gamma[0]);
  tmp = FIELD_sqr(product_coset[idx]);
  tmp = FIELD_sub(tmp, product_coset[idx]);
  tmp = FIELD_mul(tmp, l_last[idx]);
  value = FIELD_add(value, tmp);

  // (1 - (l_last(X) + l_blind(X))) * (
  //   z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
  //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
  //          (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
  // ) = 0
  value = FIELD_mul(value, y_beta_gamma[0]);
  tmp = FIELD_add(permuted_input_coset[idx], y_beta_gamma[1]);
  FIELD tmp2 = FIELD_add(permuted_table_coset[idx], y_beta_gamma[2]);
  tmp = FIELD_mul(tmp, tmp2);
  tmp = FIELD_mul(tmp, product_coset[r_next]);
  tmp2 = FIELD_mul(product_coset[idx], table[idx]);
  tmp = FIELD_sub(tmp, tmp2);
  tmp = FIELD_mul(tmp, l_active_row[idx]);
  value = FIELD_add(value, tmp);

  // l_0(X) * (a'(X) - s'(X)) = 0
  value = FIELD_mul(value, y_beta_gamma[0]);
  tmp2 = FIELD_sub(permuted_input_coset[idx], permuted_table_coset[idx]);
  tmp = FIELD_mul(tmp2, l0[idx]);
  value = FIELD_add(value, tmp);

  // (1 - (l_last + l_blind)) * (a′(X) − s′(X))⋅(a′(X) − a′(\omega^{-1} X)) = 0
  value = FIELD_mul(value, y_beta_gamma[0]);
  tmp = FIELD_sub(permuted_input_coset[idx], permuted_input_coset[r_prev]);
  tmp = FIELD_mul(tmp, tmp2);
  tmp = FIELD_mul(tmp, l_active_row[idx]);
  values[idx] = FIELD_add(value, tmp);
}


KERNEL void FIELD_eval_h_logups(
  GLOBAL FIELD* values,
  GLOBAL FIELD* table,
  GLOBAL FIELD* input_product,
  GLOBAL FIELD* input_product_sum,
  GLOBAL FIELD* m_poly_coset,
  GLOBAL FIELD* first_grand_sum_coset,
  GLOBAL FIELD* last_grand_sum_coset,
  GLOBAL FIELD* l0,
  GLOBAL FIELD* l_last,
  GLOBAL FIELD* l_active_row,
  GLOBAL FIELD* y_beta_gamma,
  uint rot,
  uint size
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;

  uint r_next = (idx + rot) & (size - 1);

  FIELD value = values[idx];

  // l_0(X) * (z_0(X)) = 0
  value = FIELD_mul(value, y_beta_gamma[0]);
  FIELD tmp = FIELD_mul(first_grand_sum_coset[idx], l0[idx]);
  value = FIELD_add(value, tmp);

  // l_last(X) * (z_l(X)) = 0
  value = FIELD_mul(value, y_beta_gamma[0]);
  tmp = FIELD_mul(last_grand_sum_coset[idx], l_last[idx]);
  value = FIELD_add(value, tmp);

  // (1 - (l_last(X) + l_blind(X))) * (
  //   τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
  //   - ∑_i τ(X) * Π_{j != i} φ_j(X) + m(X) * Π(φ_i(X))
  // ) = 0
  //=>(1 - (l_last(X) + l_blind(X))) * (
  //   (τ(X) * (ϕ(gX) - ϕ(X))+m(X))* Π(φ_i(X))
  //   - ∑_i τ(X) * Π_{j != i} φ_j(X)
  // ) = 0

  value = FIELD_mul(value, y_beta_gamma[0]);
  tmp = FIELD_sub(first_grand_sum_coset[r_next],first_grand_sum_coset[idx]);
  tmp = FIELD_mul(tmp, table[idx]);
  tmp = FIELD_add(tmp, m_poly_coset[idx]);
  tmp = FIELD_mul(tmp, input_product[idx]);
  FIELD tmp2 = FIELD_mul(table[idx],input_product_sum[idx]);
  tmp = FIELD_sub(tmp,tmp2);
  tmp = FIELD_mul(tmp, l_active_row[idx]);
  values[idx] = FIELD_add(value, tmp);
}

KERNEL void FIELD_eval_h_logups_extend(
  GLOBAL FIELD* values,
  GLOBAL FIELD* input_product,
  GLOBAL FIELD* input_product_sum,
  GLOBAL FIELD* grand_sum_coset,
  GLOBAL FIELD* l_active_row,
  GLOBAL FIELD* y_beta_gamma,
  uint rot,
  uint size
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;

  uint r_next = (idx + rot) & (size - 1);

  FIELD value = values[idx];

  // (1 - (l_last(X) + l_blind(X))) * (
  //   Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
  //   - ∑_i Π_{j != i} φ_j(X))
  // ) = 0
  value = FIELD_mul(value, y_beta_gamma[0]);
  FIELD tmp = FIELD_sub(grand_sum_coset[r_next],grand_sum_coset[idx]);
  tmp = FIELD_mul(tmp, input_product[idx]);
  tmp = FIELD_sub(tmp,input_product_sum[idx]);
  tmp = FIELD_mul(tmp, l_active_row[idx]);
  values[idx] = FIELD_add(value, tmp);
}


KERNEL void FIELD_eval_h_logup_grand_sum(
  GLOBAL FIELD* value,
  GLOBAL FIELD* curr_set,
  GLOBAL FIELD* prev_set,
  GLOBAL FIELD* l0,
  GLOBAL FIELD* y_beta_gamma,
  uint rot,
  uint size
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;

  uint r_prev = (idx + size + rot) & (size - 1);

  // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
  value[idx] = FIELD_mul(value[idx], y_beta_gamma[0]);
  FIELD tmp = FIELD_sub(curr_set[idx], prev_set[r_prev]);
  tmp = FIELD_mul(tmp, l0[idx]);
  value[idx] = FIELD_add(value[idx], tmp);
}



KERNEL void FIELD_eval_h_shuffles(
  GLOBAL FIELD* values,
  GLOBAL FIELD* input_coset,
  GLOBAL FIELD* table_coset,
  GLOBAL FIELD* product_coset,
  GLOBAL FIELD* l0,
  GLOBAL FIELD* l_last,
  GLOBAL FIELD* l_active_row,
  GLOBAL FIELD* y_beta_gamma,
  uint rot,
  uint size
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;

  uint r_next = (idx + rot) & (size - 1);

  FIELD value = values[idx];

  // l_0(X) * (1 - z(X)) = 0
  value = FIELD_mul(value, y_beta_gamma[0]);
  FIELD tmp = FIELD_sub(FIELD_ONE, product_coset[idx]);
  tmp = FIELD_mul(tmp, l0[idx]);
  value = FIELD_add(value, tmp);

  // l_last(X) * (z(X)^2 - z(X)) = 0
  value = FIELD_mul(value, y_beta_gamma[0]);
  tmp = FIELD_sqr(product_coset[idx]);
  tmp = FIELD_sub(tmp, product_coset[idx]);
  tmp = FIELD_mul(tmp, l_last[idx]);
  value = FIELD_add(value, tmp);

  // (1 - (l_last(X) + l_blind(X))) * (
  //   z(\omega X) (s(X) + \beta^i) - z(X) (a(X) + \beta^i)) = 0
  // note: beta^i calculated in input/table coset already
  value = FIELD_mul(value, y_beta_gamma[0]);
  tmp = FIELD_mul(table_coset[idx], product_coset[r_next]);
  FIELD tmp2 = FIELD_mul(input_coset[idx],product_coset[idx]);
  tmp = FIELD_sub(tmp, tmp2);
  tmp = FIELD_mul(tmp, l_active_row[idx]);

  values[idx] = FIELD_add(value, tmp);
}


KERNEL void FIELD_eval_constant(
  GLOBAL FIELD* value,
  GLOBAL FIELD* c
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;
  value[idx] =  *c;
}

KERNEL void FIELD_eval_scale(
  GLOBAL FIELD* res,
  GLOBAL FIELD* l,
  int l_rot,
  uint size,
  GLOBAL FIELD* c
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;
  uint lidx = (idx + size + l_rot) & (size - 1);
  res[idx] =  FIELD_mul(l[lidx], c[0]);
}

KERNEL void FIELD_eval_sum(
  GLOBAL FIELD* res,
  GLOBAL FIELD* l,
  GLOBAL FIELD* r,
  int l_rot,
  int r_rot,
  uint size
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;
  uint lidx = (idx + size + l_rot) & (size - 1);
  uint ridx = (idx + size + r_rot) & (size - 1);
  res[idx] =  FIELD_add(l[lidx], r[ridx]);
}

KERNEL void FIELD_eval_sum_c(
  GLOBAL FIELD* res,
  GLOBAL FIELD* l,
  int l_rot,
  GLOBAL FIELD* r,
  uint size
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;
  uint lidx = (idx + size + l_rot) & (size - 1);
  res[idx] =  FIELD_add(l[lidx], r[0]);
}

KERNEL void FIELD_eval_mul(
  GLOBAL FIELD* res,
  GLOBAL FIELD* l,
  GLOBAL FIELD* r,
  int l_rot,
  int r_rot,
  uint size
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;
  uint lidx = (idx + size + l_rot) & (size - 1);
  uint ridx = (idx + size + r_rot) & (size - 1);
  res[idx] =  FIELD_mul(l[lidx], r[ridx]);
}


KERNEL void FIELD_eval_mul_c(
  GLOBAL FIELD* res,
  GLOBAL FIELD* l,
  int l_rot,
  GLOBAL FIELD* r,
  uint size
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;
  uint lidx = (idx + size + l_rot) & (size - 1);
  res[idx] =  FIELD_mul(l[lidx], r[0]);
}

KERNEL void FIELD_eval_lcbeta(
  GLOBAL FIELD* res,
  GLOBAL FIELD* l,
  GLOBAL FIELD* r,
  int l_rot,
  int r_rot,
  uint size,
  GLOBAL FIELD* beta
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;
  uint lidx = (idx + size + l_rot) & (size - 1);
  uint ridx = (idx + size + r_rot) & (size - 1);
  res[idx] =  FIELD_mul(FIELD_add(l[lidx], beta[0]), r[ridx]);
}

KERNEL void FIELD_eval_lctheta(
  GLOBAL FIELD* res,
  GLOBAL FIELD* l,
  GLOBAL FIELD* r,
  int l_rot,
  int r_rot,
  uint size,
  GLOBAL FIELD* theta
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;
  uint lidx = (idx + size + l_rot) & (size - 1);
  uint ridx = (idx + size + r_rot) & (size - 1);
  res[idx] =  FIELD_add(FIELD_mul(l[lidx], theta[0]), r[ridx]);
}

KERNEL void FIELD_eval_addgamma(
  GLOBAL FIELD* res,
  GLOBAL FIELD* l,
  int l_rot,
  uint size,
  GLOBAL FIELD* gamma
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;
  uint lidx = (idx + size + l_rot) & (size - 1);
  res[idx] =  FIELD_add(l[lidx], gamma[0]);
}

KERNEL void FIELD_eval_fft_prepare(
  GLOBAL FIELD* origin_value,
  GLOBAL FIELD* value,
  uint origin_size
) {
  uint gid = GET_GLOBAL_ID();
  uint idx = gid;
  if (idx < origin_size) {
    value[idx] = origin_value[idx];
  } else {
    value[idx] = FIELD_ZERO;
  }
}

KERNEL void FIELD_batch_unmont(
  GLOBAL FIELD* value
) {
  uint gid = GET_GLOBAL_ID();
  uint id = gid;
  value[id] = FIELD_unmont(value[id]);
}

KERNEL void FIELD_batch_mont(
  GLOBAL FIELD* value
) {
  uint gid = GET_GLOBAL_ID();
  uint id = gid;
  value[id] = FIELD_mont(value[id]);
}

KERNEL void FIELD_sort(
  GLOBAL FIELD* value,
  uint i,
  uint j
) {
  uint gid = GET_GLOBAL_ID();
  uint id = gid;

  uint direction = (id >> i) & 1;
  uint log_step = i - j;
  uint step = 1 << log_step;
  uint offset = id & (step - 1);
  uint index0 = ((id >> log_step) << (log_step + 1)) | offset;
  uint index1 = index0 | step;

  if (FIELD_gte(value[index0], value[index1]) ^ direction)
  {
    FIELD t = value[index0];
    value[index0] = value[index1];
    value[index1] = t;
  }
}

KERNEL void FIELD_calc_lookup_z(
  GLOBAL FIELD* permuted_input,
  GLOBAL FIELD* permuted_table,
  GLOBAL FIELD* compressed_input,
  GLOBAL FIELD* compressed_table,
  GLOBAL FIELD* beta_gamma,
  uint len,
  uint slot_len)
{
  uint gid = GET_GLOBAL_ID();
  uint id = gid;
  uint start = id * slot_len;
  uint end = (start + slot_len) > len ? len : (start + slot_len);

  if (start < end) {
    FIELD beta = beta_gamma[0];
    FIELD gamma = beta_gamma[1];
    FIELD t1;
    FIELD t2;
    FIELD t3 = FIELD_ONE;

    for (uint i = start; i < end; i++) {
      t1 = FIELD_add(permuted_input[i], beta);
      t2 = FIELD_add(permuted_table[i], gamma);
      t1 = FIELD_mul(t1, t2);
      permuted_input[i] = t3;
      permuted_table[i] = t1;
      t3 = FIELD_mul(t1, t3);
    }

    FIELD to_inv = t3;
    t3 = FIELD_ONE;

    for (uint j = 0; j < end - start; j++) {
      uint i = end - j - 1;
      t1 = FIELD_add(compressed_input[i], beta);
      t2 = FIELD_add(compressed_table[i], gamma);
      t1 = FIELD_mul(t1, t2);
      t1 = FIELD_mul(t1, permuted_input[i]);
      t1 = FIELD_mul(t1, t3);
      t3 = FIELD_mul(permuted_table[i], t3);

      permuted_input[i] = t1;
    }

    permuted_table[start] = to_inv;
  }
}
