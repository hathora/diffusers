#!/bin/bash

# Test runner script for Triton kernels
# Usage: ./run_tests.sh [options]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include current directory for imports
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}Triton Kernel Test Suite${NC}"
echo -e "${GREEN}====================================${NC}\n"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}pytest not found. Installing test dependencies...${NC}"
    pip install -r requirements_test.txt
fi

# Default: run all non-benchmark tests
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Running all tests (excluding benchmarks)...${NC}\n"
    pytest test_kernels.py -v -m "not benchmark"
    exit 0
fi

# Handle different test modes
case "$1" in
    quick)
        echo -e "${YELLOW}Running quick sanity checks...${NC}\n"
        python3 test_kernels.py
        ;;
    all)
        echo -e "${YELLOW}Running all tests including benchmarks...${NC}\n"
        pytest test_kernels.py -v
        ;;
    benchmark)
        echo -e "${YELLOW}Running benchmark tests only...${NC}\n"
        pytest test_kernels.py -v -m benchmark
        ;;
    coverage)
        echo -e "${YELLOW}Running tests with coverage...${NC}\n"
        pytest test_kernels.py -v -m "not benchmark" --cov=flux_triton_ops --cov-report=html --cov-report=term
        echo -e "\n${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    rope)
        echo -e "${YELLOW}Running RoPE tests only...${NC}\n"
        pytest test_kernels.py -v -k "rope"
        ;;
    rms)
        echo -e "${YELLOW}Running RMSNorm tests only...${NC}\n"
        pytest test_kernels.py -v -k "rms_norm"
        ;;
    ln)
        echo -e "${YELLOW}Running LayerNorm tests only...${NC}\n"
        pytest test_kernels.py -v -k "layer_norm"
        ;;
    geglu)
        echo -e "${YELLOW}Running GeGLU tests only...${NC}\n"
        pytest test_kernels.py -v -k "geglu"
        ;;
    parallel)
        echo -e "${YELLOW}Running tests in parallel...${NC}\n"
        pytest test_kernels.py -v -n auto -m "not benchmark"
        ;;
    help)
        echo "Usage: ./run_tests.sh [mode]"
        echo ""
        echo "Modes:"
        echo "  quick       - Run quick sanity checks (no pytest)"
        echo "  all         - Run all tests including benchmarks"
        echo "  benchmark   - Run only benchmark tests"
        echo "  coverage    - Run tests with coverage report"
        echo "  rope        - Run only RoPE tests"
        echo "  rms         - Run only RMSNorm tests"
        echo "  ln          - Run only LayerNorm tests"
        echo "  geglu       - Run only GeGLU tests"
        echo "  parallel    - Run tests in parallel"
        echo "  help        - Show this help message"
        echo ""
        echo "Default (no args): Run all non-benchmark tests"
        ;;
    *)
        echo -e "${RED}Unknown mode: $1${NC}"
        echo "Run './run_tests.sh help' for usage information"
        exit 1
        ;;
esac

echo -e "\n${GREEN}====================================${NC}"
echo -e "${GREEN}Tests completed!${NC}"
echo -e "${GREEN}====================================${NC}"

