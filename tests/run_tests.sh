#!/bin/bash

# script for testing the Python certifier.
# Supports running two kinds of tests:
# "all" - here, we run the certifier over all of the test inputs of a model (e.g., 10000 MNIST test images).
#         we check whether the real-arithmetic Python certifier agrees with the Dafny reference implementation
#
# "cex" - here, we run the certifier over counter-examples to the Dafny certifier
#         we check that the Python certifier does not certify any counter-examples, but that the real-arithmetic Python
#         certifier would have certified all of them

CERTIFIER=../robust_certifier.py

MNIST_RESULTS_GRAM_11="results_epsilon_0.45_[128,128,128,128,128,128,128,128]_500_eval_0.3_gram_11.json"
MNIST_RESULTS_GRAM_20="results_epsilon_0.45_[128,128,128,128,128,128,128,128]_500_eval_0.3_gram_20.json"
FASHION_MNIST_RESULTS_GRAM_12="results_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500_eval_0.25_gram_12.json"
FASHION_MNIST_RESULTS_GRAM_13="results_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500_eval_0.25_gram_13.json"
CIFAR10_RESULTS_GRAM_12="results_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800_eval_0.141_gram_12.json"

MNIST_NEURAL_NET="neural_net_mnist_epsilon_0.45_[128,128,128,128,128,128,128,128]_500.txt"
FASHION_MNIST_NEURAL_NET="neural_net_mnist_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500.txt"
CIFAR10_NEURAL_NET="neural_net_mnist_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800.txt"

ALL_MNIST_TEST_INPUTS="all_mnist_test_inputs/test_inputs_epsilon_0.3.json"
ALL_FASHION_MNIST_TEST_INPUTS="all_fashion_mnist_test_inputs/test_inputs_epsilon_0.25.json"
ALL_CIFAR10_TEST_INPUTS="all_cifar10_test_inputs/all_test_inputs.json"

CEX_MNIST_FLOAT32="cex_mnist_deepfool/counter_examples.json"
CEX_MNIST_FLOAT16="cex_mnist_deepfool_float16/counter_examples.json"
CEX_MNIST_FLOAT64="cex_mnist_deepfool_float64/counter_examples.json"

CEX_FASHION_MNIST_FLOAT32="cex_fashion_mnist_deepfool/counter_examples.json"

CEX_CIFAR10_FLOAT32="cex_cifar10_deepfool/counter_examples.json"

function run_test() {
    format="$1"                # float32, float16, etc.
    model="$2"                 # mnist, fashion_mnist, etc.
    gram_iters="$3"
    kind="$4"                  # cex or all

    # variables to set: $nn_file, $ref_results_file, $cex_file
    case $model in
	"mnist")
	    nn_file="${MNIST_NEURAL_NET}"
	    case $gram_iters in
		"20")
		    ref_results_file="${MNIST_RESULTS_GRAM_20}"
		    ;;
		"11")
		    ref_results_file="${MNIST_RESULTS_GRAM_11}"
		    ;;
		*)
		    echo "Unrecognised gram iters for model $model"
		    exit 1
		    ;;
	    esac
	    case $kind in
		"all")
		    if [ "$format" != "float32" ]; then
			echo "Only float32 format supported when running kind all"
			exit 1
		    fi
		    cex_file="${ALL_MNIST_TEST_INPUTS}"
		    ;;
		"cex")
		    case $format in
			"float32")
			    cex_file="${CEX_MNIST_FLOAT32}"
			    ;;
			"float16")
			    cex_file="${CEX_MNIST_FLOAT16}"
			    ;;
			"float64")
			    cex_file="${CEX_MNIST_FLOAT64}"
			    ;;
			*)
			    echo "Unsupported format for model $model and kind $kind"
			    exit 1
			    ;;
		    esac
		    ;;
		*)
		    echo "Unrecognised kind $kind"
		    exit 1
		    ;;
	    esac
	    ;;
	"fashion_mnist")
	    nn_file="${FASHION_MNIST_NEURAL_NET}"
	    case $gram_iters in
		"12")
		    ref_results_file="${FASHION_MNIST_RESULTS_GRAM_12}"
		    ;;
		"13")
		    ref_results_file="${FASHION_MNIST_RESULTS_GRAM_13}"
		    ;;
		*)
		    echo "Unrecognised gram iters for model $model"
		    exit 1
		    ;;
	    esac
	    case $kind in
		"all")
		    if [ "$format" != "float32" ]; then
			echo "Only float32 format supported when running kind all"
			exit 1
		    fi
		    cex_file="${ALL_FASHION_MNIST_TEST_INPUTS}"
		    ;;
		"cex")
		    case $format in
			"float32")
			    cex_file="${CEX_FASHION_MNIST_FLOAT32}"
			    ;;
			*)
			    echo "Unsupported format for model $model and kind $kind"
			    exit 1
			    ;;
		    esac
		    ;;
		*)
		    echo "Unrecognised kind $kind"
		    exit 1
		    ;;
	    esac
	    ;;
	"cifar10")
	    nn_file="${CIFAR10_NEURAL_NET}"
	    case $gram_iters in
		"12")
		    ref_results_file="${CIFAR10_RESULTS_GRAM_12}"
		    ;;
		*)
		    echo "Unrecognised gram iters for model $model"
		    exit 1
		    ;;
	    esac
	    case $kind in
		"all")
		    if [ "$format" != "float32" ]; then
			echo "Only float32 format supported when running kind all"
			exit 1
		    fi
		    cex_file="${ALL_CIFAR10_TEST_INPUTS}"
		    ;;
		"cex")
		    case $format in
			"float32")
			    cex_file="${CEX_CIFAR10_FLOAT32}"
			    ;;
			*)
			    echo "Unsupported format for model $model and kind $kind"
			    exit 1
			    ;;
		    esac
		    ;;
		*)
		    echo "Unrecognised kind $kind"
		    exit 1
		    ;;
	    esac
	    ;;
	*)
	    echo "Unrecognised model"
	    exit 1
	    ;;
    esac

    # check we set everything
    if [ "$nn_file" == "" ]; then
	echo "Internal error: no nn_file set"
	exit 1
    fi
    if [ "$ref_results_file" == "" ]; then
	echo "Internal error: no ref_results_file set"
	exit 1
    fi
    if [ "$cex_file" == "" ]; then
	echo "Internal error: no cex_file set"
	exit 1
    fi

    echo -n "Running test: $format, $model, $gram_iters, $kind ...  "
    python "${CERTIFIER}" "$format" "$nn_file" "$gram_iters" --cex "$cex_file" "$ref_results_file"  > .log 2>&1
    count=$(cat .log | grep '^Got\ [0-9]\+\ instances\ to\ certify...' | cut -d' ' -f2)
    count_ok=$(cat .log | grep '^Certified\ [0-9]\+\ instances\ as\ robust' | cut -d' ' -f2)
    count_failed=$(cat .log | grep '^Failed\ to\ certify\ [0-9]\+\ instances\ as\ robust' | cut -d' ' -f4)
    count_ok_real=$(cat .log | grep 'Real\ certifier\ would\ have\ certified\ [0-9]\+\ instances\ as\ robust' | cut -d' ' -f6)

    if (( $count_ok + $count_failed != $count )); then
	echo "ERROR: Internal error: counts don't add up!"
	exit 1
    fi

    if [ "$kind" == "cex" ]; then
	if (( $count_ok != 0 )); then
	    echo "ERROR: Certifier certified $count_ok counter-examples!"
	    exit 1
	fi
	if (( $count_ok_real != $count )); then
	    echo "ERROR: Real-arithmetic certifier would not have certified all counter-examples!"
	    exit 1
	fi
    fi
    if [ "$kind" == "all" ]; then
	ref_num=$(cat "${ref_results_file}" | grep true | wc -l)
	if (( $count_ok_real != $ref_num )); then
	    echo "ERROR: Real-arithmetic certifier $count_ok doesn't agree with Dafny reference $ref_num!"
	    exit 1
	fi
    fi
    echo "OK"
}

run_test "float32" "mnist" 20 "cex"
run_test "float16" "mnist" 20 "cex"
run_test "float64" "mnist" 20 "cex"

run_test "float32" "fashion_mnist" 13 "cex"

run_test "float32" "mnist" 11 "all"
run_test "float32" "mnist" 20 "all"
run_test "float32" "fashion_mnist" 12 "all"
run_test "float32" "fashion_mnist" 13 "all"
