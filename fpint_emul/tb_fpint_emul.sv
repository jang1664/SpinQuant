`timescale 1ns / 1ps

`include "VX_define.vh"

import cf_math_pkg::*;
import fpint_emul::*;

module tb_fpint_emul();
  task automatic test_fpint_gemm_qcol_2scomp(
    input int M, int K, int N,
    input int mode=0,
    bit DEBUG=0
  );

    logic [IN_WIDTH-1:0] input_data[MAX_M*MAX_K];
    logic [MAX_W_WIDTH-1:0] weight_data[MAX_K*MAX_N];
    logic [S_WIDTH-1:0] scale_data[MAX_K*MAX_NG];
    logic [Z_WIDTH-1:0] zero_data[MAX_K*MAX_NG];
    logic [O_WIDTH-1:0] output_data[MAX_M*MAX_N];
    logic [O_WIDTH-1:0] output_data_ref[MAX_M*MAX_N];
    int fail_cnt = 0;

    $display("===============================================");
    $display("FPINT GEMM QROW 2SCOMP TEST: M=%0d K=%0d N=%0d mode=%0d", M, K, N, mode);
    $display("===============================================");

    // Initialize input data
    for(int m=0; m<M; m++) begin
      for(int k=0; k<K; k++) begin
        if(mode == 0) begin
          input_data[m*K + k] = fp32_val_to_fp16_bit((signed'($urandom_range(0, 100))-50)/10.0);
        end else begin
          input_data[m*K + k] = fp32_val_to_fp16_bit(1.0);
        end
      end
    end

    for(int k=0; k<K; k++) begin
      for(int n=0; n<N; n++) begin
        if(mode == 0) begin
          weight_data[k*N + n] = (signed'($urandom_range(0, 15)) - 8);
        end else begin
          weight_data[k*N + n] = 1;
        end
      end
    end

    for(int k=0; k<K; k++) begin
      for(int ng=0; ng<(N+QBLOCK-1)/QBLOCK; ng++) begin
        if(mode == 0) begin
          scale_data[k*(N/QBLOCK) + ng] = fp32_val_to_fp16_bit((signed'($urandom_range(0, 100))-50)/10.0);
          zero_data[k*(N/QBLOCK) + ng]  = signed'($urandom_range(0, 15)) - 8;
        end else begin
          scale_data[k*(N/QBLOCK) + ng] = fp32_val_to_fp16_bit(1.0);
          zero_data[k*(N/QBLOCK) + ng]  = -1;
        end
      end
    end

    // Call the DUT function
    fpint_emul::fpint_gemm_qcol_2scomp(
      input_data,
      weight_data,
      scale_data,
      zero_data,
      M, N, K,
      output_data,
      DEBUG
    );

    // Call ref function
    fpint_emul::fpint_gemm_ref(
      input_data,
      weight_data,
      scale_data,
      zero_data,
      M, N, K,
      output_data_ref,
      QCOL,
      WNOTRANS,
      DEBUG
    );

    // Display output data
    for(int m=0; m<M; m++) begin
      for(int n=0; n<N; n++) begin
        shortreal out_val = fp16_bit_to_fp16_val(output_data[m*N + n]);
        shortreal ref_val = fp16_bit_to_fp16_val(output_data_ref[m*N + n]);
        real rel_err = rel_err_fp32(out_val, ref_val);
        $display("Output[%0d, %0d]: DUT=%h, REF=%h | rel_err=%f", m, n, output_data[m*N + n], output_data_ref[m*N + n], rel_err);
        if (rel_err > 0.01) begin
          fail_cnt = fail_cnt + 1;
          $error("Mismatch at Output[%0d, %0d]: DUT=%h, REF=%h", m, n, output_data[m*N + n], output_data_ref[m*N + n]);
        end
      end
    end

    if(fail_cnt > 0) begin
      $error("TEST FAILED. %d/%d", fail_cnt, M*N);
    end else begin
      $display("TEST PASSED");
    end
  endtask

  task automatic test_fpint_gemm_qcol_zero_less(
    input int M, int K, int N,
    input int mode=0,
    bit DEBUG=0
  );

    logic [IN_WIDTH-1:0] input_data[MAX_M*MAX_K];
    logic [MAX_W_WIDTH-1:0] weight_data[MAX_K*MAX_N];
    logic [S_WIDTH-1:0] scale_data[MAX_KG*MAX_N];
    logic [Z_WIDTH-1:0] zero_data[MAX_KG*MAX_N];
    logic [O_WIDTH-1:0] output_data[MAX_M*MAX_N];
    logic [O_WIDTH-1:0] output_data_ref[MAX_M*MAX_N];
    int fail_cnt = 0;

    $display("===============================================");
    $display("FPINT GEMM QCOL ZERO LESS TEST: M=%0d K=%0d N=%0d mode=%0d", M, K, N, mode);
    $display("===============================================");

    // Initialize input data
    for(int m=0; m<M; m++) begin
      for(int k=0; k<K; k++) begin
        if(mode == 0) begin
          input_data[m*K + k] = fp32_val_to_fp16_bit((signed'($urandom_range(0, 100))-50)/10.0);
        end else begin
          input_data[m*K + k] = fp32_val_to_fp16_bit(1.0);
        end
      end
    end

    for(int k=0; k<K; k++) begin
      for(int n=0; n<N; n++) begin
        if(mode == 0) begin
          weight_data[k*N + n] = 2*((signed'($urandom_range(0, 15)) - 8))+1;
        end else begin
          weight_data[k*N + n] = 1;
        end
      end
    end

    for(int kg=0; kg<(K+QBLOCK-1)/QBLOCK; kg++) begin
      for(int n=0; n<N; n++) begin
        if(mode == 0) begin
          scale_data[kg*N + n] = fp32_val_to_fp16_bit((signed'($urandom_range(0, 100))-50)/10.0);
          zero_data[kg*N + n]  = signed'($urandom_range(0, 15)) - 8;
        end else begin
          scale_data[kg*N + n] = fp32_val_to_fp16_bit(1.0);
          zero_data[kg*N + n]  = -1;
        end
      end
    end

    // Call the DUT function
    fpint_emul::fpint_gemm_qcol_zero_less(
      input_data,
      weight_data,
      scale_data,
      zero_data,
      M, N, K,
      output_data,
      DEBUG
    );

    // Call ref function
    fpint_emul::fpint_gemm_ref(
      input_data,
      weight_data,
      scale_data,
      zero_data,
      M, N, K,
      output_data_ref,
      QCOL,
      WNOTRANS,
      DEBUG
    );

    // Display output data
    for(int m=0; m<M; m++) begin
      for(int n=0; n<N; n++) begin
        shortreal out_val = fp16_bit_to_fp16_val(output_data[m*N + n]);
        shortreal ref_val = fp16_bit_to_fp16_val(output_data_ref[m*N + n]);
        real rel_err = rel_err_fp32(out_val, ref_val);
        $display("Output[%0d, %0d]: DUT=%h, REF=%h | rel_err=%f", m, n, output_data[m*N + n], output_data_ref[m*N + n], rel_err);
        if (rel_err > 0.01) begin
          fail_cnt = fail_cnt + 1;
          $error("Mismatch at Output[%0d, %0d]: DUT=%h, REF=%h", m, n, output_data[m*N + n], output_data_ref[m*N + n]);
        end
      end
    end

    if(fail_cnt > 0) begin
      $error("TEST FAILED. %d/%d", fail_cnt, M*N);
    end else begin
      $display("TEST PASSED");
    end
  endtask

  task automatic test_fpint_gemm_qrow_2scomp(
    input int M, int K, int N,
    input int mode=0,
    bit DEBUG=0
  );

    logic [IN_WIDTH-1:0] input_data[MAX_M*MAX_K];
    logic [MAX_W_WIDTH-1:0] weight_data[MAX_K*MAX_N];
    logic [S_WIDTH-1:0] scale_data[MAX_K*MAX_NG];
    logic [Z_WIDTH-1:0] zero_data[MAX_K*MAX_NG];
    logic [O_WIDTH-1:0] output_data[MAX_M*MAX_N];
    logic [O_WIDTH-1:0] output_data_ref[MAX_M*MAX_N];
    int fail_cnt = 0;

    $display("===============================================");
    $display("FPINT GEMM QROW 2SCOMP TEST: M=%0d K=%0d N=%0d mode=%0d", M, K, N, mode);
    $display("===============================================");

    // Initialize input data
    for(int m=0; m<M; m++) begin
      for(int k=0; k<K; k++) begin
        automatic int rand_val;
        automatic shortreal fp_val;
        if(mode == 0) begin
          rand_val = signed'($urandom_range(0, 100)) - 50;
          fp_val = shortreal'(rand_val) / 10.0;
          input_data[m*K + k] = fp32_val_to_fp16_bit(fp_val);
        end else begin
          // rand_val = signed'($urandom_range(0, 100)) - 50;
          // fp_val = shortreal'(rand_val) / 10.0;
          // input_data[m*K + k] = fp32_val_to_fp16_bit(fp_val);
          input_data[m*K + k] = fp32_val_to_fp16_bit(1.0);
        end
      end
    end

    for(int k=0; k<K; k++) begin
      for(int n=0; n<N; n++) begin
        if(mode == 0) begin
          weight_data[k*N + n] = (signed'($urandom_range(0, 15)) - 8);
        end else begin
          weight_data[k*N + n] = 1;
          // weight_data[k*N + n] = (signed'($urandom_range(0, 15)) - 8);
        end
      end
    end

    for(int k=0; k<K; k++) begin
      for(int ng=0; ng<(N/QBLOCK); ng++) begin
        if(mode == 0) begin
          scale_data[k*(N/QBLOCK) + ng] = fp32_val_to_fp16_bit((signed'($urandom_range(0, 100))-50)/10.0);
          zero_data[k*(N/QBLOCK) + ng]  = signed'($urandom_range(0, 15)) - 8;
        end else begin
          scale_data[k*(N/QBLOCK) + ng] = fp32_val_to_fp16_bit(1.0);
          zero_data[k*(N/QBLOCK) + ng]  = -1;
        end
      end
    end

    // Call the DUT function
    fpint_emul::fpint_gemm_qrow_2scomp(
      input_data,
      weight_data,
      scale_data,
      zero_data,
      M, N, K,
      output_data,
      DEBUG
    );

    // Call ref function
    fpint_emul::fpint_gemm_ref(
      input_data,
      weight_data,
      scale_data,
      zero_data,
      M, N, K,
      output_data_ref,
      QROW,
      WNOTRANS,
      DEBUG
    );

    // Display output data
    for(int m=0; m<M; m++) begin
      for(int n=0; n<N; n++) begin
        shortreal out_val = fp16_bit_to_fp16_val(output_data[m*N + n]);
        shortreal ref_val = fp16_bit_to_fp16_val(output_data_ref[m*N + n]);
        real rel_err = rel_err_fp32(out_val, ref_val);
        $display("Output[%0d, %0d]: DUT=%h, REF=%h | rel_err=%f", m, n, output_data[m*N + n], output_data_ref[m*N + n], rel_err);
        if (abs_real(rel_err) > 0.01) begin
          fail_cnt = fail_cnt + 1;
          $error("Mismatch at Output[%0d, %0d]: DUT=%h, REF=%h", m, n, output_data[m*N + n], output_data_ref[m*N + n]);
        end
      end
    end

    if(fail_cnt > 0) begin
      $error("TEST FAILED. %d/%d", fail_cnt, M*N);
    end else begin
      $display("TEST PASSED");
    end
  endtask

  task automatic test_fpint_gemm_qrow_zero_less(
    input int M, int K, int N,
    input int mode=0,
    bit DEBUG=0
  );

    logic [IN_WIDTH-1:0] input_data[MAX_M*MAX_K];
    logic [MAX_W_WIDTH-1:0] weight_data[MAX_K*MAX_N];
    logic [S_WIDTH-1:0] scale_data[MAX_K*MAX_NG];
    logic [Z_WIDTH-1:0] zero_data[MAX_K*MAX_NG];
    logic [O_WIDTH-1:0] output_data[MAX_M*MAX_N];
    logic [O_WIDTH-1:0] output_data_ref[MAX_M*MAX_N];
    int fail_cnt = 0;

    $display("===============================================");
    $display("FPINT GEMM QROW ZERO LESS TEST: M=%0d K=%0d N=%0d mode=%0d", M, K, N, mode);
    $display("===============================================");

    // Initialize input data
    for(int m=0; m<M; m++) begin
      for(int k=0; k<K; k++) begin
        automatic int rand_val;
        automatic shortreal fp_val;
        if(mode == 0) begin
          rand_val = signed'($urandom_range(0, 100)) - 50;
          fp_val = shortreal'(rand_val) / 10.0;
          input_data[m*K + k] = fp32_val_to_fp16_bit(fp_val);
        end else begin
          // rand_val = signed'($urandom_range(0, 100)) - 50;
          // fp_val = shortreal'(rand_val) / 10.0;
          // input_data[m*K + k] = fp32_val_to_fp16_bit(fp_val);
          input_data[m*K + k] = fp32_val_to_fp16_bit(1.0);
        end
      end
    end

    for(int k=0; k<K; k++) begin
      for(int n=0; n<N; n++) begin
        if(mode == 0) begin
          weight_data[k*N + n] = 2*(signed'($urandom_range(0, 15)) - 8) + 1;
        end else begin
          weight_data[k*N + n] = 1;
          // weight_data[k*N + n] = (signed'($urandom_range(0, 15)) - 8);
        end
      end
    end

    for(int k=0; k<K; k++) begin
      for(int ng=0; ng<(N/QBLOCK); ng++) begin
        if(mode == 0) begin
          scale_data[k*(N/QBLOCK) + ng] = fp32_val_to_fp16_bit((signed'($urandom_range(0, 100))-50)/10.0);
          zero_data[k*(N/QBLOCK) + ng]  = signed'($urandom_range(0, 15)) - 8;
        end else begin
          scale_data[k*(N/QBLOCK) + ng] = fp32_val_to_fp16_bit(1.0);
          zero_data[k*(N/QBLOCK) + ng]  = -1;
        end
      end
    end

    // Call the DUT function
    fpint_emul::fpint_gemm_qrow_zero_less(
      input_data,
      weight_data,
      scale_data,
      zero_data,
      M, N, K,
      output_data,
      DEBUG
    );

    // Call ref function
    fpint_emul::fpint_gemm_ref(
      input_data,
      weight_data,
      scale_data,
      zero_data,
      M, N, K,
      output_data_ref,
      QROW,
      WNOTRANS,
      DEBUG
    );

    // Display output data
    for(int m=0; m<M; m++) begin
      for(int n=0; n<N; n++) begin
        shortreal out_val = fp16_bit_to_fp16_val(output_data[m*N + n]);
        shortreal ref_val = fp16_bit_to_fp16_val(output_data_ref[m*N + n]);
        real rel_err = rel_err_fp32(out_val, ref_val);
        $display("Output[%0d, %0d]: DUT=%h, REF=%h | rel_err=%f", m, n, output_data[m*N + n], output_data_ref[m*N + n], rel_err);
        if (abs_real(rel_err) > 0.01) begin
          fail_cnt = fail_cnt + 1;
          $error("Mismatch at Output[%0d, %0d]: DUT=%h, REF=%h", m, n, output_data[m*N + n], output_data_ref[m*N + n]);
        end
      end
    end

    if(fail_cnt > 0) begin
      $error("TEST FAILED. %d/%d", fail_cnt, M*N);
    end else begin
      $display("TEST PASSED");
    end
  endtask

  task automatic fpint_gemm_qrow_real_2scomp(
    input int M, int K, int N,
    input int mode=0,
    bit DEBUG=0
  );

    logic [IN_WIDTH-1:0] input_data[MAX_M*MAX_K];
    logic [MAX_W_WIDTH-1:0] weight_data[MAX_K*MAX_N];
    logic [S_WIDTH-1:0] scale_data[MAX_K*MAX_NG];
    logic [Z_WIDTH-1:0] zero_data[MAX_K*MAX_NG];
    logic [O_WIDTH-1:0] output_data[MAX_M*MAX_N];
    logic [O_WIDTH-1:0] output_data_ref[MAX_M*MAX_N];
    int fail_cnt = 0;

    $display("===============================================");
    $display("FPINT GEMM QROW REAL 2SCOMP TEST: M=%0d K=%0d N=%0d mode=%0d", M, K, N, mode);
    $display("===============================================");

    // Initialize input data
    for(int m=0; m<M; m++) begin
      for(int k=0; k<K; k++) begin
        automatic int rand_val;
        automatic shortreal fp_val;
        if(mode == 0) begin
          rand_val = signed'($urandom_range(0, 100)) - 50;
          fp_val = shortreal'(rand_val) / 10.0;
          input_data[m*K + k] = fp32_val_to_fp16_bit(fp_val);
        end else begin
          // rand_val = signed'($urandom_range(0, 100)) - 50;
          // fp_val = shortreal'(rand_val) / 10.0;
          // input_data[m*K + k] = fp32_val_to_fp16_bit(fp_val);
          input_data[m*K + k] = fp32_val_to_fp16_bit(1.0);
        end
      end
    end

    for(int k=0; k<K; k++) begin
      for(int n=0; n<N; n++) begin
        if(mode == 0) begin
          weight_data[k*N + n] = (signed'($urandom_range(0, 15)) - 8);
        end else begin
          weight_data[k*N + n] = 1;
          // weight_data[k*N + n] = (signed'($urandom_range(0, 15)) - 8);
        end
      end
    end

    for(int k=0; k<K; k++) begin
      for(int ng=0; ng<(N/QBLOCK); ng++) begin
        if(mode == 0) begin
          scale_data[k*(N/QBLOCK) + ng] = fp32_val_to_fp16_bit((signed'($urandom_range(0, 100))-50)/10.0);
          zero_data[k*(N/QBLOCK) + ng]  = signed'($urandom_range(0, 15)) - 8;
        end else begin
          scale_data[k*(N/QBLOCK) + ng] = fp32_val_to_fp16_bit(1.0);
          zero_data[k*(N/QBLOCK) + ng]  = -1;
        end
      end
    end

    // Call the DUT function
    fpint_emul::fpint_gemm_qrow_real_2scomp(
      input_data,
      weight_data,
      scale_data,
      zero_data,
      M, N, K,
      output_data,
      DEBUG
    );

    // Call ref function
    fpint_emul::fpint_gemm_ref(
      input_data,
      weight_data,
      scale_data,
      zero_data,
      M, N, K,
      output_data_ref,
      QROW,
      WNOTRANS,
      DEBUG
    );

    // Display output data
    for(int m=0; m<M; m++) begin
      for(int n=0; n<N; n++) begin
        shortreal out_val = fp16_bit_to_fp16_val(output_data[m*N + n]);
        shortreal ref_val = fp16_bit_to_fp16_val(output_data_ref[m*N + n]);
        real rel_err = rel_err_fp32(out_val, ref_val);
        $display("Output[%0d, %0d]: DUT=%h, REF=%h | rel_err=%f", m, n, output_data[m*N + n], output_data_ref[m*N + n], rel_err);
        if (abs_real(rel_err) > 0.01) begin
          fail_cnt = fail_cnt + 1;
          $error("Mismatch at Output[%0d, %0d]: DUT=%h, REF=%h", m, n, output_data[m*N + n], output_data_ref[m*N + n]);
        end
      end
    end

    if(fail_cnt > 0) begin
      $error("TEST FAILED. %d/%d", fail_cnt, M*N);
    end else begin
      $display("TEST PASSED");
    end
  endtask

  initial begin
    test_fpint_gemm_qcol_2scomp(32, 128, 128);
    test_fpint_gemm_qcol_zero_less(32, 128, 128);
    test_fpint_gemm_qrow_2scomp(32, 128, 128, 0, 0);
    test_fpint_gemm_qrow_zero_less(32, 128, 128, 0, 0);
    fpint_gemm_qrow_real_2scomp(32, 128, 128, 0, 0);
    $finish;
  end

endmodule