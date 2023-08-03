#include "MaxFlops_half.h"

int main() {
  intilizeDeviceProp(0);

  fp16_max_flops();

  return 1;
}