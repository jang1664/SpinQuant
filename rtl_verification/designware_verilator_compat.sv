`timescale 1ns/1ps

// Minimal simulation-only compatibility models for the DesignWare cells used
// by gemm_unit_wrap_opt_v3.  The floating-point helpers are implemented in
// fp32_dpi.cpp so Verilator performs IEEE-754 binary32 operations rather than
// promoting SystemVerilog shortreal values to real.

import "DPI-C" function int unsigned spinquant_fp32_mul_bits(
    input int unsigned a,
    input int unsigned b
);
import "DPI-C" function int unsigned spinquant_fp32_add_bits(
    input int unsigned a,
    input int unsigned b
);
import "DPI-C" function int unsigned spinquant_i32_to_fp32_bits(
    input int signed a
);
import "DPI-C" function int unsigned spinquant_bf16_mul_bits(
    input int unsigned a,
    input int unsigned b
);
import "DPI-C" function int unsigned spinquant_i32_to_bf16_bits(
    input int signed a
);

module DW_lzd #(
    parameter int a_width = 8
) (
    input  logic [a_width-1:0] a,
    output logic [a_width-1:0] dec,
    output logic [$clog2(a_width):0] enc
);
  integer i;
  logic found;
  always_comb begin
    dec = '0;
    enc = a_width;
    found = 1'b0;
    for (i = a_width - 1; i >= 0; i = i - 1) begin
      if (!found && a[i]) begin
        enc = a_width - 1 - i;
        dec[i] = 1'b1;
        found = 1'b1;
      end
    end
  end
endmodule

module DW_fifo_s1_sf #(
    parameter int width = 8,
    parameter int depth = 2,
    parameter int rst_mode = 3
) (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             push_req_n,
    input  logic             pop_req_n,
    input  logic             diag_n,
    input  logic [width-1:0] data_in,
    output logic             empty,
    output logic             almost_empty,
    output logic             half_full,
    output logic             almost_full,
    output logic             full,
    output logic             error,
    output logic [width-1:0] data_out
);
  localparam int PTR_W = (depth <= 2) ? 1 : $clog2(depth);
  logic [width-1:0] mem [0:depth-1];
  logic [PTR_W-1:0] rd_ptr;
  logic [PTR_W-1:0] wr_ptr;
  integer count;
  wire push = ~push_req_n;
  wire pop = ~pop_req_n;

  always_comb begin
    empty = (count == 0);
    full = (count == depth);
    almost_empty = (count <= 1);
    half_full = (count * 2 >= depth);
    almost_full = (count >= depth - 1);
    error = (push && full && !pop) || (pop && empty && !push);
    data_out = empty ? '0 : mem[rd_ptr];
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_ptr <= '0;
      wr_ptr <= '0;
      count <= 0;
      for (integer j = 0; j < depth; j = j + 1) mem[j] <= '0;
    end else begin
      if (push && (!full || pop)) begin
        mem[wr_ptr] <= data_in;
        wr_ptr <= (wr_ptr == depth - 1) ? '0 : wr_ptr + 1'b1;
      end
      if (pop && (!empty || push)) begin
        rd_ptr <= (rd_ptr == depth - 1) ? '0 : rd_ptr + 1'b1;
      end
      case ({push && (!full || pop), pop && (!empty || push)})
        2'b10: count <= count + 1;
        2'b01: count <= count - 1;
        default: count <= count;
      endcase
    end
  end

  wire unused_diag = diag_n;
  wire [31:0] unused_rst_mode = rst_mode;
endmodule

module DW_fp_mult #(
    parameter int sig_width = 23,
    parameter int exp_width = 8,
    parameter int ieee_compliance = 1
) (
    input  logic [sig_width+exp_width:0] a,
    input  logic [sig_width+exp_width:0] b,
    input  logic [2:0]                   rnd,
    output logic [sig_width+exp_width:0] z,
    output logic [7:0]                   status
);
  generate
    if (sig_width == 23 && exp_width == 8) begin : g_fp32
      always_comb z = spinquant_fp32_mul_bits(a, b);
    end else if (sig_width == 7 && exp_width == 8) begin : g_bf16
      always_comb z = spinquant_bf16_mul_bits(a, b);
    end else begin : g_unsupported
      always_comb z = '0;
    end
  endgenerate
  always_comb status = '0;
  wire [2:0] unused_rnd = rnd;
  wire [31:0] unused_ieee = ieee_compliance;
endmodule

module DW_fp_add #(
    parameter int sig_width = 23,
    parameter int exp_width = 8,
    parameter int ieee_compliance = 1
) (
    input  logic [sig_width+exp_width:0] a,
    input  logic [sig_width+exp_width:0] b,
    input  logic [2:0]                   rnd,
    output logic [sig_width+exp_width:0] z,
    output logic [7:0]                   status
);
  generate
    if (sig_width == 23 && exp_width == 8) begin : g_fp32
      always_comb z = spinquant_fp32_add_bits(a, b);
    end else begin : g_unsupported
      always_comb z = '0;
    end
  endgenerate
  always_comb status = '0;
  wire [2:0] unused_rnd = rnd;
  wire [31:0] unused_ieee = ieee_compliance;
endmodule

module DW_fp_i2flt #(
    parameter int sig_width = 23,
    parameter int exp_width = 8,
    parameter int isize = 32,
    parameter int isign = 1
) (
    input  logic [isize-1:0]             a,
    input  logic [2:0]                   rnd,
    output logic [sig_width+exp_width:0] z,
    output logic [7:0]                   status
);
  generate
    if (sig_width == 23 && exp_width == 8 && isize == 32) begin : g_fp32
      always_comb z = spinquant_i32_to_fp32_bits($signed(a));
    end else if (sig_width == 7 && exp_width == 8 && isize == 32) begin : g_bf16
      always_comb z = spinquant_i32_to_bf16_bits($signed(a));
    end else begin : g_unsupported
      always_comb z = '0;
    end
  endgenerate
  always_comb status = '0;
  wire [2:0] unused_rnd = rnd;
  wire [31:0] unused_isign = isign;
endmodule
