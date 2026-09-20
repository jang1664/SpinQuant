`timescale 1ns/1ps
`include "common_define.svh"
import common_parameters::*;
import common_types::*;

module tb_gemm_unit_wrap_opt_v3_verilator;
  localparam int MAX_DW = 32;
  localparam int IN_DW = 4;
  localparam int S_IN_DW = 5;
  localparam int OUT_DW = 32;
  localparam int WT_SUM_DW = IN_DW + $clog2(PE_ROW);
  localparam int AB_ADDR_W = $clog2(AB_DEPTH);
  localparam int MAX_M = 256;

  logic clk_i = 1'b0;
  logic resetn_i;
  logic mode_set_i;
  in_mode_t in_mode_i;
  w_mode_t w_mode_i;
  logic [PE_ROW*MAX_DW-1:0] data_i;
  logic valid_i;
  logic ready_o;
  logic [PE_COL-1:0][3:0] weight_i;
  logic [PE_COL-1:0][IN_DW-1:0] weight_bar_i;
  logic [PE_COL-1:0][WT_SUM_DW-1:0] wt_sum_i;
  logic in_weight_sel_i;
  logic out_weight_sel_i;
  logic ready_weight_i;
  logic load_i;
  logic dequant_fp32_i;
  logic [PE_COL-1:0][MAX_DW-1:0] dequant_scale_i;
  logic [PE_COL-1:0][MAX_DW-1:0] acc_mem_rd_data_i;
  logic [PE_COL-1:0][AB_ADDR_W-1:0] acc_mem_rd_addr_o;
  logic [PE_COL-1:0] acc_mem_rd_en_o;
  logic [PE_COL-1:0][AB_ADDR_W-1:0] acc_mem_wr_addr_o;
  logic [PE_COL-1:0] acc_mem_wr_en_o;
  logic [PE_COL-1:0][MAX_DW-1:0] acc_mem_wr_data_o;
  logic [PE_COL*MAX_DW-1:0] data_o;
  logic [PE_COL-1:0] valid_o;
  logic [PE_COL-1:0][MAX_DW-1:0] acc_data_o;
  logic [PE_COL-1:0] acc_valid_o;

  logic [15:0] activation_mem [0:MAX_M*PE_ROW-1];
  logic [7:0] weight_mem [0:PE_ROW-1];
  logic [31:0] scale_mem [0:PE_COL-1];
  integer result_count [0:PE_COL-1];
  integer m_rows;
  string activation_file;
  string weight_file;
  string scale_file;
  string mode_name;
  logic trace_enabled;

  always #2ns clk_i = ~clk_i;

  gemm_unit_wrap_opt_v3 #(
      .MAX_DW(MAX_DW),
      .IN_DW(IN_DW),
      .S_IN_DW(S_IN_DW),
      .OUT_DW(OUT_DW)
  ) u_dut (
      .clk_i(clk_i),
      .resetn_i(resetn_i),
      .mode_set_i(mode_set_i),
      .in_mode_i(in_mode_i),
      .w_mode_i(w_mode_i),
      .data_i(data_i),
      .valid_i(valid_i),
      .ready_o(ready_o),
      .weight_i(weight_i),
      .weight_bar_i(weight_bar_i),
      .wt_sum_i(wt_sum_i),
      .in_weight_sel_i(in_weight_sel_i),
      .out_weight_sel_i(out_weight_sel_i),
      .ready_weight_i(ready_weight_i),
      .load_i(load_i),
      .dequant_fp32_i(dequant_fp32_i),
      .dequant_scale_i(dequant_scale_i),
      .acc_mem_rd_data_i(acc_mem_rd_data_i),
      .acc_mem_rd_addr_o(acc_mem_rd_addr_o),
      .acc_mem_rd_en_o(acc_mem_rd_en_o),
      .acc_mem_wr_addr_o(acc_mem_wr_addr_o),
      .acc_mem_wr_en_o(acc_mem_wr_en_o),
      .acc_mem_wr_data_o(acc_mem_wr_data_o),
      .data_o(data_o),
      .valid_o(valid_o),
      .acc_data_o(acc_data_o),
      .acc_valid_o(acc_valid_o)
  );

  function automatic logic [3:0] raw_weight(input int row, input int col);
    raw_weight = weight_mem[row][4*col+:4];
  endfunction

  function automatic logic [3:0] recoded_weight(input logic [3:0] raw);
    recoded_weight = raw ^ 4'h8;
  endfunction

  function automatic logic [WT_SUM_DW-1:0] weight_sum(input int col);
    integer sum;
    begin
      sum = 0;
      for (integer row = 0; row < PE_ROW; row = row + 1) begin
        sum = sum + recoded_weight(raw_weight(row, col));
      end
      weight_sum = sum[WT_SUM_DW-1:0];
    end
  endfunction

  task automatic drive_defaults;
    begin
      resetn_i = 1'b0;
      mode_set_i = 1'b0;
      in_mode_i = I_BF16;
      w_mode_i = W_INT4;
      data_i = '0;
      valid_i = 1'b0;
      weight_i = '0;
      weight_bar_i = '0;
      wt_sum_i = '0;
      in_weight_sel_i = 1'b0;
      out_weight_sel_i = 1'b0;
      ready_weight_i = 1'b0;
      load_i = 1'b1;
      dequant_fp32_i = 1'b1;
      acc_mem_rd_data_i = '0;
      for (integer c = 0; c < PE_COL; c = c + 1) begin
        dequant_scale_i[c] = scale_mem[c];
        result_count[c] = 0;
      end
    end
  endtask

  task automatic load_weights;
    logic [3:0] raw;
    begin
      for (integer cycle = 0; cycle < PE_ROW; cycle = cycle + 1) begin
        @(negedge clk_i);
        for (integer col = 0; col < PE_COL; col = col + 1) begin
          raw = raw_weight(PE_ROW - 1 - cycle, col);
          weight_i[col] = raw;
          weight_bar_i[col] = recoded_weight(raw);
          wt_sum_i[col] = weight_sum(col);
        end
        ready_weight_i = 1'b1;
      end
      @(negedge clk_i);
      ready_weight_i = 1'b0;
      weight_i = '0;
      weight_bar_i = '0;
      wt_sum_i = '0;
    end
  endtask

  task automatic stream_row(input int row_index);
    begin
      while (!ready_o) @(negedge clk_i);
      data_i = '0;
      for (integer r = 0; r < PE_ROW; r = r + 1) begin
        data_i[16*r+:16] = activation_mem[row_index*PE_ROW+r];
      end
      valid_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      valid_i = 1'b0;
      data_i = '0;
    end
  endtask

  always @(posedge clk_i) begin
    #1ps;
    if (resetn_i) begin
      for (integer c = 0; c < PE_COL; c = c + 1) begin
        if (valid_o[c]) begin
          $display("RTL_RESULT mode=%s row=%0d col=%0d fp32=%08x out16=%04x",
                   mode_name, result_count[c], c, acc_data_o[c], data_o[16*c+:16]);
          result_count[c] = result_count[c] + 1;
        end
      end
    end
  end

  always @(posedge clk_i) begin
    #1ps;
    if (trace_enabled && u_dut.u_gemm_unit.mxu_valid_o[0]) begin
      $display("RTL_TRACE_MXU ps=%08x sext=%0d chunk=%0d ifmap0=%x",
               u_dut.u_gemm_unit.u_mxu_ps_o[0],
               u_dut.u_gemm_unit.inpt_sign_ext_i[0],
               u_dut.u_gemm_unit.chunk_idx_at_ifmap,
               u_dut.u_gemm_unit.dsu_data_o[0][0]);
    end
    if (trace_enabled && u_dut.u_gemm_unit.cm_valid_o[0]) begin
      $display("RTL_TRACE cm=%012x max_exp=%02x abs=%012x enc=%02x fp=%08x",
               u_dut.u_gemm_unit.cm_ps_o[0],
               u_dut.u_gemm_unit.dsu_max_exp_o,
               u_dut.u_gemm_unit.u_int2fp_array.col[0].u_int2fp_wide.abs_int,
               u_dut.u_gemm_unit.u_int2fp_array.col[0].u_int2fp_wide.enc,
               u_dut.u_gemm_unit.u_int2fp_array.fp_data_wide[0]);
    end
  end

  initial begin
    integer timeout_cycles;
    if (!$value$plusargs("ACT=%s", activation_file)) $fatal(1, "missing +ACT=<file>");
    if (!$value$plusargs("WEIGHT=%s", weight_file)) $fatal(1, "missing +WEIGHT=<file>");
    if (!$value$plusargs("SCALE=%s", scale_file)) $fatal(1, "missing +SCALE=<file>");
    if (!$value$plusargs("MODE=%s", mode_name)) $fatal(1, "missing +MODE=bf16|fp16");
    if (!$value$plusargs("M=%d", m_rows)) $fatal(1, "missing +M=<rows>");
    trace_enabled = $test$plusargs("TRACE");
    if (m_rows <= 0 || m_rows > MAX_M) $fatal(1, "M outside supported range");

    $readmemh(activation_file, activation_mem);
    $readmemh(weight_file, weight_mem);
    $readmemh(scale_file, scale_mem);
    drive_defaults();
    if (mode_name == "bf16") in_mode_i = I_BF16;
    else if (mode_name == "fp16") in_mode_i = I_FP16;
    else $fatal(1, "unsupported mode %s", mode_name);

    repeat (3) @(posedge clk_i);
    @(negedge clk_i);
    resetn_i = 1'b1;
    @(negedge clk_i);
    mode_set_i = 1'b1;
    @(negedge clk_i);
    mode_set_i = 1'b0;

    load_weights();
    if (trace_enabled) begin
      $display("RTL_TRACE_WEIGHT raw0=%x raw1=%x raw127=%x mem0=%x mem1=%x mem127=%x sum=%0d",
               raw_weight(0, 0), raw_weight(1, 0), raw_weight(127, 0),
               u_dut.u_gemm_unit.u_mxu.weight_bar_mem[0][0][0],
               u_dut.u_gemm_unit.u_mxu.weight_bar_mem[0][1][0],
               u_dut.u_gemm_unit.u_mxu.weight_bar_mem[0][127][0],
               u_dut.u_gemm_unit.u_mxu.weight_bar_sum_mem[0][0]);
    end
    for (integer m = 0; m < m_rows; m = m + 1) stream_row(m);

    timeout_cycles = 0;
    while ((result_count[0] < m_rows || result_count[1] < m_rows) &&
           timeout_cycles < 4096) begin
      @(posedge clk_i);
      timeout_cycles = timeout_cycles + 1;
    end
    if (result_count[0] != m_rows || result_count[1] != m_rows) begin
      $fatal(1, "output timeout: got col0=%0d col1=%0d expected=%0d",
             result_count[0], result_count[1], m_rows);
    end
    $display("RTL_DONE mode=%s rows=%0d outputs=%0d", mode_name, m_rows,
             result_count[0] + result_count[1]);
    $finish;
  end
endmodule
