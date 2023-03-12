#pragma once

#include <functional>
#include <tuple>

#include "../matrix/matrix.cuh"
#include "../util/img.cuh"
#include "activations.cuh"

template<typename T, std::size_t INPUT_SIZE, std::size_t HIDDEN_SIZE, std::size_t OUTPUT_SIZE>
class NeuralNetwork {
public:
	NeuralNetwork() {
		this->hidden_weights.randomize(HIDDEN_SIZE);
		this->output_weights.randomize(OUTPUT_SIZE);
	}

	NeuralNetwork(std::ifstream& in): hidden_weights(in), output_weights(in) { }

	std::tuple<
		Matrix<T, HIDDEN_SIZE, INPUT_SIZE>,
		Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>
	> train(
		const Vector<T, INPUT_SIZE>& input,
		const Vector<T, OUTPUT_SIZE>& expected_output,
		std::function<T(const T&)>& activation,
		std::function<T(const T&)>& activation_prime
	) const {
		Vector<T, HIDDEN_SIZE> hidden_output;
		Vector<T, OUTPUT_SIZE> final_output;
		std::tie(hidden_output, final_output) = feed_forward(input, activation);
		Vector<T, HIDDEN_SIZE> hidden_errors;
		Vector<T, OUTPUT_SIZE> output_errors;
		std::tie(hidden_errors, output_errors) = find_errors(expected_output, final_output);
		return back_propagate(hidden_errors, output_errors, hidden_output, final_output, input, activation_prime);
	}

	std::tuple<
		Matrix<T, HIDDEN_SIZE, INPUT_SIZE>,
		Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>
	> train_mini_batch(
		Img* imgs,
		std::function<T(const T&)>& activation,
		std::function<T(const T&)>& activation_prime,
		std::size_t mini_batch_size
	) const {
		Matrix<T, HIDDEN_SIZE, INPUT_SIZE> hidden_delta_avg(0);
		Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE> output_delta_avg(0);
		for(size_t i = 0; i < mini_batch_size; i++) {
			Img* cur_img = imgs + i;
			Vector<T, OUTPUT_SIZE> expected_output(0);
			expected_output[cur_img->label] = 1;

			Matrix<T, HIDDEN_SIZE, INPUT_SIZE> hidden_delta;
			Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE> output_delta;

			std::tie(hidden_delta, output_delta) = train(cur_img->img_data, expected_output, activation, activation_prime);

			hidden_delta_avg += hidden_delta;
			output_delta_avg += output_delta;
		}

		return std::make_tuple(hidden_delta_avg, output_delta_avg);
	}

	template<std::size_t BATCH_SIZE>
	static void train_mini_batch(
		CudaArray<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>, BATCH_SIZE>& output_delta_buff,
		CudaArray<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>, BATCH_SIZE>& hidden_delta_buff,
		CudaArray<Vector<T, INPUT_SIZE>, BATCH_SIZE>& inputs_buff,
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& expected_outputs_buff,
		CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE>& hidden_output_buff,
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& final_output_buff,
		CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE>& hidden_errors_buff,
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& final_errors_buff,
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& multiplication_buff_output,
		CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE>& multiplication_buff_hidden,
		CudaBox<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>>& hidden_weights,
		CudaBox<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>>& output_weights,
		CudaBox<Matrix<T, HIDDEN_SIZE, OUTPUT_SIZE>>& transposed_output_weights_buff,
		const CudaBox<T>& coef,
		const Img* imgs,
		ActivationCuda::ActivationFunction activation,
		int maxThreadsPerBlock
	) {
		KernelParams pt(HIDDEN_SIZE * OUTPUT_SIZE, maxThreadsPerBlock);
		MatrixCuda::transpose<<<pt.bc, pt.tc>>>(&transposed_output_weights_buff, &output_weights);

		Vector<T, INPUT_SIZE> inputs[BATCH_SIZE];
		Vector<T, OUTPUT_SIZE> expected_outputs[BATCH_SIZE];
		for(size_t i = 0; i < BATCH_SIZE; i++) {
			const Img* cur_img = imgs + i;
			Vector<T, OUTPUT_SIZE> expected_output(0);
			expected_output[cur_img->label] = 1;
			expected_outputs[i] = expected_output;
			inputs[i] = cur_img->img_data;
		}
		inputs_buff.put(inputs);
		expected_outputs_buff.put(expected_outputs);

		CHECK_SYNC;

		train(
			hidden_delta_buff, 
			output_delta_buff, 
			hidden_output_buff, 
			final_output_buff,
			hidden_errors_buff,
			final_errors_buff,
			multiplication_buff_output, 
			multiplication_buff_hidden,
			hidden_weights,
			output_weights,
			transposed_output_weights_buff,
			inputs_buff,
			expected_outputs_buff,
			activation,
			maxThreadsPerBlock
		);

		KernelParams pro(BATCH_SIZE * HIDDEN_SIZE * OUTPUT_SIZE, maxThreadsPerBlock);
		KernelParams prh(BATCH_SIZE * INPUT_SIZE * HIDDEN_SIZE, maxThreadsPerBlock);

		MatrixCuda::reduce_mul_add<<<pro.bc, pro.tc>>>(&output_weights, &output_delta_buff, &coef);
		MatrixCuda::reduce_mul_add<<<prh.bc, prh.tc>>>(&hidden_weights, &hidden_delta_buff, &coef);
		CHECK_SYNC;
	}

	template<std::size_t BATCH_SIZE>
	void train_batch_inner(
		CudaBox<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>>& output_delta,
		CudaBox<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>>& hidden_delta,
		CudaArray<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>, BATCH_SIZE>& output_delta_buff,
		CudaArray<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>, BATCH_SIZE>& hidden_delta_buff,
		CudaArray<Vector<T, INPUT_SIZE>, BATCH_SIZE>& inputs_buff,
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& expected_outputs_buff,
		CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE>& hidden_output_buff,
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& final_output_buff,
		CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE>& hidden_errors_buff,
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& final_errors_buff,
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& multiplication_buff_output,
		CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE>& multiplication_buff_hidden,
		CudaBox<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>>& hidden_weights,
		CudaBox<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>>& output_weights,
		CudaBox<Matrix<T, HIDDEN_SIZE, OUTPUT_SIZE>>& transposed_output_weights_buff,
		CudaBox<T>& coef_buff,
		Img* imgs, 
		T lr,
		ActivationCuda::ActivationFunction activation,
		int maxThreadsPerBlock
	) {

		T coef = lr / BATCH_SIZE;
		coef_buff.put(&coef);

		train_mini_batch(
			output_delta_buff,
			hidden_delta_buff,
			inputs_buff,
			expected_outputs_buff,
			hidden_output_buff,
			final_output_buff,
			hidden_errors_buff,
			final_errors_buff,
			multiplication_buff_output,
			multiplication_buff_hidden,
			hidden_weights,
			output_weights,
			transposed_output_weights_buff,
			coef_buff,
			imgs,
			activation,
			maxThreadsPerBlock
		);
	}

	template<std::size_t BATCH_SIZE>
	void train_batch_cuda(
		Img* imgs,
		std::size_t epochs,
		std::size_t batch_size,
		T lr,
		T lr_coef,
		ActivationCuda::ActivationFunction activation
	) {
		CudaBox<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>> output_delta;
		CudaBox<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>> hidden_delta;
		CudaArray<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>, BATCH_SIZE> output_delta_buff;
		CudaArray<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>, BATCH_SIZE> hidden_delta_buff;
		CudaArray<Vector<T, INPUT_SIZE>, BATCH_SIZE> inputs_buff;
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE> expected_outputs_buff;
		CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE> hidden_output_buff;
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE> final_output_buff;
		CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE> hidden_errors_buff;
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE> final_errors_buff;
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE> multiplication_buff_output;
		CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE> multiplication_buff_hidden;
		CudaBox<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>> hidden_weights(this->hidden_weights);
		CudaBox<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>> output_weights(this->output_weights);
		CudaBox<Matrix<T, HIDDEN_SIZE, OUTPUT_SIZE>> transposed_output_weights_buff;
		CudaBox<T> coef_buff;
		int maxThreadsPerBlock = KernelParams::getMaxThreadsPerBlock();
		for(std::size_t e = 1; e <= epochs; e++) {
			for (std::size_t i = 0; i < batch_size; i += BATCH_SIZE) {
				std::cout << "(CUDA) Epoch " << e << '/' << epochs << ", Img Batch No. " << ((i / BATCH_SIZE) + 1) << '/' << (batch_size / BATCH_SIZE) << std::endl;
				train_batch_inner(
					output_delta,
					hidden_delta,
					output_delta_buff,
					hidden_delta_buff,
					inputs_buff,
					expected_outputs_buff,
					hidden_output_buff,
					final_output_buff,
					hidden_errors_buff,
					final_errors_buff,
					multiplication_buff_output,
					multiplication_buff_hidden,
					hidden_weights,
					output_weights,
					transposed_output_weights_buff,
					coef_buff,
					imgs + i, 
					lr, 
					activation,
					maxThreadsPerBlock
				);
			}
			lr *= lr_coef;
		}
		this->hidden_weights = hidden_weights.to_host();
		this->output_weights = output_weights.to_host();
	}

	void train_batch_inner(
		Img* imgs, 
		T lr,
		std::function<T(const T&)>& activation,
		std::function<T(const T&)>& activation_prime,
		std::size_t mini_batch_size
	) {
		Matrix<T, HIDDEN_SIZE, INPUT_SIZE> hidden_delta_avg;
		Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE> output_delta_avg;

		std::tie(hidden_delta_avg, output_delta_avg) = train_mini_batch(imgs, activation, activation_prime, mini_batch_size);

		this->hidden_weights += hidden_delta_avg * (lr / mini_batch_size);
		this->output_weights += output_delta_avg * (lr / mini_batch_size);
	}

	void train_batch(
		Img* imgs,
		std::size_t epochs,
		std::size_t batch_size,
		std::size_t mini_batch_size,
		T lr,
		const T& lr_coef,
		std::function<T(const T&)>& activation,
		std::function<T(const T&)>& activation_prime
	) {
		for(std::size_t e = 1; e <= epochs; e++) {
			for (std::size_t i = 0; i < batch_size; i += mini_batch_size) {
				std::cout << "Epoch " << e << '/' << epochs << ", Img Batch No. " << (i / mini_batch_size) + 1 << '/' << batch_size / mini_batch_size << std::endl;
				train_batch_inner(imgs + i, lr, activation, activation_prime, mini_batch_size);
			}
			lr *= lr_coef;
		}
	}

	Vector<T, OUTPUT_SIZE> predict(const Vector<T, INPUT_SIZE>& input, std::function<T(const T&)>& activation) const {
		auto feed = feed_forward(input, activation);
		auto res = std::get<1>(feed);
		return res.softmax();
	}

	std::size_t predict_img(const Img& img, std::function<T(const T&)>& activation) const {
		Vector<T, OUTPUT_SIZE> res = predict(img.img_data, activation);
		return res.argmax();
	}

	double predict_imgs(Img* imgs, std::size_t n_imgs, std::function<T(const T&)>& activation) const {
		std::size_t n_correct = 0;
		for(std::size_t i = 0; i < n_imgs; i++) {
			Img& img = imgs[i];
			std::size_t prediction = predict_img(img, activation);
			if(prediction == img.label) {
				n_correct++;
			}
		}
		return 1.0 * n_correct / n_imgs;
	}

	std::ofstream& save_binary(std::ofstream& out) const {
		this->hidden_weights.save_binary(out);
		this->output_weights.save_binary(out);
		return out;
	}


private:

	std::tuple<Vector<T, HIDDEN_SIZE>, Vector<T, OUTPUT_SIZE>> feed_forward(
		const Vector<T, INPUT_SIZE>& input, 
		std::function<T(const T&)>& activation
	) const {
		Vector<T, HIDDEN_SIZE> hidden_output_unactivated = this->hidden_weights.dot(input);
		Vector<T, HIDDEN_SIZE> hidden_output = hidden_output_unactivated.apply(activation);
		Vector<T, OUTPUT_SIZE> final_output_unactivated = this->output_weights.dot(hidden_output);
		Vector<T, OUTPUT_SIZE> final_output = final_output_unactivated.apply(activation);
		return std::make_tuple(hidden_output, final_output);
	}


	std::tuple<Vector<T, HIDDEN_SIZE>, Vector<T, OUTPUT_SIZE>> find_errors(
		const Vector<T, OUTPUT_SIZE>& expected_output, 
		const Vector<T, OUTPUT_SIZE>& final_output
	) const {
		Matrix<T, HIDDEN_SIZE, OUTPUT_SIZE> transposed_mat = this->output_weights.transpose();
		Vector<T, OUTPUT_SIZE> output_errors = expected_output - final_output;
		Vector<T, HIDDEN_SIZE> hidden_errors = transposed_mat.dot(output_errors);
		return std::make_tuple(hidden_errors, output_errors);
	}

	template<std::size_t WEIGHTS_ROWS, std::size_t WEIGHTS_COLS> 
	static Matrix<T, WEIGHTS_ROWS, WEIGHTS_COLS> back_propagate_core(
		const Vector<T, WEIGHTS_ROWS>& output, 
		const Vector<T, WEIGHTS_ROWS>& errors,
		const Vector<T, WEIGHTS_COLS>& input,
		std::function<T(const T&)>& activation_prime
	) {
		Vector<T, WEIGHTS_ROWS> primed_output = output.apply(activation_prime);
		primed_output *= errors;
		return primed_output.dot(input);
	}

	static std::tuple<
		Matrix<T, HIDDEN_SIZE, INPUT_SIZE>,
		Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>
	> back_propagate(
		const Vector<T, HIDDEN_SIZE>& hidden_errors, 
		const Vector<T, OUTPUT_SIZE>& output_errors,
		const Vector<T, HIDDEN_SIZE>& hidden_output, 
		const Vector<T, OUTPUT_SIZE>& final_output,
		const Vector<T, INPUT_SIZE>& input,
		std::function<T(const T&)>& activation_prime
	) {
		Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE> output_delta = back_propagate_core(
			final_output, 
			output_errors, 
			hidden_output, 
			activation_prime
		);
		Matrix<T, HIDDEN_SIZE, INPUT_SIZE> hidden_delta = back_propagate_core(
			hidden_output, 
			hidden_errors, 
			input, 
			activation_prime
		);

		return std::make_tuple(hidden_delta, output_delta);
	}

	// CUDA IMPLEMENTATIONS:

	static void feed_forward(
		CudaBox<Vector<T, HIDDEN_SIZE>>& hidden_output,
		CudaBox<Vector<T, OUTPUT_SIZE>>& final_output,
		const CudaBox<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>>& hidden_weights,
		const CudaBox<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>>& output_weights,
		const CudaBox<Vector<T, INPUT_SIZE>>& input, 
		ActivationCuda::ActivationFunction activation
	) {
		MatrixCuda::dot<<<1, HIDDEN_SIZE>>>(&hidden_output, &hidden_weights, &input);
		CHECK_SYNC;
		switch(activation) {
			case ActivationCuda::Relu:
				ActivationCuda::apply_relu<<<1, HIDDEN_SIZE>>>(&hidden_output);
				break;
			case ActivationCuda::Sigmoid:
				ActivationCuda::apply_sigmoid<<<1, HIDDEN_SIZE>>>(&hidden_output);
				break;
			default:
				break;
		}
		CHECK_SYNC;
		MatrixCuda::dot<<<1, OUTPUT_SIZE>>>(&final_output, &output_weights, &hidden_output);
		CHECK_SYNC;
		switch(activation) {
			case ActivationCuda::Relu:
				ActivationCuda::apply_relu<<<1, OUTPUT_SIZE>>>(&final_output);
				break;
			case ActivationCuda::Sigmoid:
				ActivationCuda::apply_sigmoid<<<1, OUTPUT_SIZE>>>(&final_output);
				break;
			default:
				break;
		}
		CHECK_SYNC;
	}

	

	template<std::size_t BATCH_SIZE>
	static void feed_forward(
		CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE>& hidden_output,
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& final_output,
		const CudaBox<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>>& hidden_weights,
		const CudaBox<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>>& output_weights,
		const CudaArray<Vector<T, INPUT_SIZE>, BATCH_SIZE>& input,
		ActivationCuda::ActivationFunction activation,
		int maxThreadsPerBlock
	) {
		KernelParams pdi(BATCH_SIZE * HIDDEN_SIZE, maxThreadsPerBlock);
		switch(activation) {
			case ActivationCuda::Relu:
				ActivationCuda::dot_relu<<<pdi.bc, pdi.tc>>>(&hidden_output, &hidden_weights, &input);
				break;
			case ActivationCuda::Sigmoid:
				ActivationCuda::dot_sigmoid<<<pdi.bc, pdi.tc>>>(&hidden_output, &hidden_weights, &input);
				break;
			default:
				break;
		}
		KernelParams pdo(BATCH_SIZE * OUTPUT_SIZE, maxThreadsPerBlock);
		CHECK_SYNC;
		switch(activation) {
			case ActivationCuda::Relu:
				ActivationCuda::dot_relu<<<pdo.bc, pdo.tc>>>(&final_output, &output_weights, &hidden_output);
				break;
			case ActivationCuda::Sigmoid:
				ActivationCuda::dot_sigmoid<<<pdo.bc, pdo.tc>>>(&final_output, &output_weights, &hidden_output);
				break;
			default:
				break;
		}
		CHECK_SYNC;
	}

	static void find_errors(
		CudaBox<Vector<T, HIDDEN_SIZE>>& hidden_errors,
		CudaBox<Vector<T, OUTPUT_SIZE>>& output_errors,
		const CudaBox<Matrix<T, HIDDEN_SIZE, OUTPUT_SIZE>>& transposed_output_weights,
		const CudaBox<Vector<T, OUTPUT_SIZE>>& expected_output, 
		const CudaBox<Vector<T, OUTPUT_SIZE>>& final_output
	) {
		VectorCuda::sub<<<1, OUTPUT_SIZE>>>(&output_errors, &expected_output, &final_output);
		CHECK_SYNC;
		MatrixCuda::dot<<<HIDDEN_SIZE, OUTPUT_SIZE>>>(&hidden_errors, &transposed_output_weights, &output_errors);
		CHECK_SYNC;
	}

	template<std::size_t BATCH_SIZE>
	static void find_errors(
		CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE>& hidden_errors,
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& output_errors,
		const CudaBox<Matrix<T, HIDDEN_SIZE, OUTPUT_SIZE>>& transposed_output_weights,
		const CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& expected_output, 
		const CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& final_output,
		int maxThreadsPerBlock
	) {
		KernelParams ps(BATCH_SIZE * OUTPUT_SIZE, maxThreadsPerBlock);
		VectorCuda::sub<<<ps.bc, ps.tc>>>(&output_errors, &expected_output, &final_output);
		KernelParams pd(BATCH_SIZE * HIDDEN_SIZE * OUTPUT_SIZE, maxThreadsPerBlock);
		CHECK_SYNC;
		MatrixCuda::dot<<<pd.bc, pd.tc>>>(&hidden_errors, &transposed_output_weights, &output_errors);
		CHECK_SYNC;
	}

	template<std::size_t WEIGHTS_ROWS, std::size_t WEIGHTS_COLS> 
	static void back_propagate_core(
		CudaBox<Matrix<T, WEIGHTS_ROWS, WEIGHTS_COLS>>& delta,
		CudaBox<Vector<T, WEIGHTS_ROWS>>& multiplication_buff,
		const CudaBox<Vector<T, WEIGHTS_ROWS>>& output, 
		const CudaBox<Vector<T, WEIGHTS_ROWS>>& errors,
		const CudaBox<Vector<T, WEIGHTS_COLS>>& input,
		ActivationCuda::ActivationFunction activation_prime
	) {
		switch(activation_prime) {
			case ActivationCuda::Relu:
				ActivationCuda::apply_relu_prime<<<1, WEIGHTS_ROWS>>>(&multiplication_buff, &output);
				break;
			case ActivationCuda::Sigmoid:
				ActivationCuda::apply_sigmoid_prime<<<1, WEIGHTS_ROWS>>>(&multiplication_buff, &output);
				break;
			default:
				break;
		}
		CHECK_SYNC;
		VectorCuda::mul<<<1, WEIGHTS_ROWS>>>(&multiplication_buff, &errors);
		CHECK_SYNC;
		VectorCuda::dot<<<WEIGHTS_ROWS, WEIGHTS_COLS>>>(&delta, &multiplication_buff, &input);
		CHECK_SYNC;
	}

	template<std::size_t WEIGHTS_ROWS, std::size_t WEIGHTS_COLS, std::size_t BATCH_SIZE> 
	static void back_propagate_core(
		CudaArray<Matrix<T, WEIGHTS_ROWS, WEIGHTS_COLS>, BATCH_SIZE>& delta,
		CudaArray<Vector<T, WEIGHTS_ROWS>, BATCH_SIZE>& multiplication_buff,
		const CudaArray<Vector<T, WEIGHTS_ROWS>, BATCH_SIZE>& output, 
		const CudaArray<Vector<T, WEIGHTS_ROWS>, BATCH_SIZE>& errors,
		const CudaArray<Vector<T, WEIGHTS_COLS>, BATCH_SIZE>& input,
		ActivationCuda::ActivationFunction activation_prime,
		int maxThreadsPerBlock
	) {
		KernelParams pa(BATCH_SIZE * WEIGHTS_ROWS, maxThreadsPerBlock);
		switch(activation_prime) {
			case ActivationCuda::Relu:
				ActivationCuda::apply_relu_prime_mul<<<pa.bc, pa.tc>>>(&multiplication_buff, &output, &errors);
				break;
			case ActivationCuda::Sigmoid:
				ActivationCuda::apply_sigmoid_prime_mul<<<pa.bc, pa.tc>>>(&multiplication_buff, &output, &errors);
				break;
			default:
				break;
		}
		KernelParams pd(BATCH_SIZE * WEIGHTS_ROWS * WEIGHTS_COLS, maxThreadsPerBlock);
		CHECK_SYNC;
		VectorCuda::dot<<<pd.bc, pd.tc>>>(&delta, &multiplication_buff, &input);
		CHECK_SYNC;
	}

	static void back_propagate(
		CudaBox<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>>& output_delta,
		CudaBox<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>>& hidden_delta,
		CudaBox<Vector<T, OUTPUT_SIZE>>& multiplication_buff_output,
		CudaBox<Vector<T, HIDDEN_SIZE>>& multiplication_buff_hidden,
		const CudaBox<Vector<T, HIDDEN_SIZE>>& hidden_errors, 
		const CudaBox<Vector<T, OUTPUT_SIZE>>& output_errors,
		const CudaBox<Vector<T, HIDDEN_SIZE>>& hidden_output, 
		const CudaBox<Vector<T, OUTPUT_SIZE>>& final_output,
		const CudaBox<Vector<T, INPUT_SIZE>>& input,
		ActivationCuda::ActivationFunction activation_prime
	) {
		back_propagate_core(
			output_delta,
			multiplication_buff_output,
			final_output, 
			output_errors, 
			hidden_output, 
			activation_prime
		);
		back_propagate_core(
			hidden_delta,
			multiplication_buff_hidden,
			hidden_output, 
			hidden_errors, 
			input, 
			activation_prime
		);
	}

	template<std::size_t BATCH_SIZE>
	static void back_propagate(
		CudaArray<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>, BATCH_SIZE>& output_delta,
		CudaArray<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>, BATCH_SIZE>& hidden_delta,
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& multiplication_buff_output,
		CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE>& multiplication_buff_hidden,
		const CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE>& hidden_errors, 
		const CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& output_errors,
		const CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE>& hidden_output, 
		const CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& final_output,
		const CudaArray<Vector<T, INPUT_SIZE>, BATCH_SIZE>& input,
		ActivationCuda::ActivationFunction activation_prime,
		int maxThreadsPerBlock
	) {
		back_propagate_core(
			output_delta,
			multiplication_buff_output,
			final_output, 
			output_errors, 
			hidden_output, 
			activation_prime,
			maxThreadsPerBlock
		);
		back_propagate_core(
			hidden_delta,
			multiplication_buff_hidden,
			hidden_output, 
			hidden_errors, 
			input, 
			activation_prime,
			maxThreadsPerBlock
		);
	}

	static void train(
		CudaBox<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>>& output_delta,
		CudaBox<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>>& hidden_delta,
		CudaBox<Vector<T, HIDDEN_SIZE>>& hidden_output_buff,
		CudaBox<Vector<T, OUTPUT_SIZE>>& final_output_buff,
		CudaBox<Vector<T, HIDDEN_SIZE>>& hidden_errors_buff,
		CudaBox<Vector<T, OUTPUT_SIZE>>& output_errors_buff,
		CudaBox<Vector<T, OUTPUT_SIZE>>& multiplication_buff_output,
		CudaBox<Vector<T, HIDDEN_SIZE>>& multiplication_buff_hidden,
		const CudaBox<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>>& hidden_weights,
		const CudaBox<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>>& output_weights,
		const CudaBox<Matrix<T, HIDDEN_SIZE, OUTPUT_SIZE>>& transposed_output_weights,
		const CudaBox<Vector<T, INPUT_SIZE>>& input,
		const CudaBox<Vector<T, OUTPUT_SIZE>>& expected_output,
		ActivationCuda::ActivationFunction activation
	) {
		feed_forward(hidden_output_buff, final_output_buff, hidden_weights, output_weights, input, activation);
		find_errors(hidden_errors_buff, output_errors_buff, transposed_output_weights, expected_output, final_output_buff);
		back_propagate(output_delta, hidden_delta, multiplication_buff_output, multiplication_buff_hidden, hidden_errors_buff, output_errors_buff, hidden_output_buff, final_output_buff, input, activation);
	}

	template<std::size_t BATCH_SIZE>
	static void train(
		CudaArray<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>, BATCH_SIZE>& hidden_delta,
		CudaArray<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>, BATCH_SIZE>& output_delta,
		CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE>& hidden_output_buff,
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& final_output_buff,
		CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE>& hidden_errors_buff,
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& output_errors_buff,
		CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& multiplication_buff_output,
		CudaArray<Vector<T, HIDDEN_SIZE>, BATCH_SIZE>& multiplication_buff_hidden,
		const CudaBox<Matrix<T, HIDDEN_SIZE, INPUT_SIZE>>& hidden_weights,
		const CudaBox<Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>>& output_weights,
		const CudaBox<Matrix<T, HIDDEN_SIZE, OUTPUT_SIZE>>& transposed_output_weights,
		const CudaArray<Vector<T, INPUT_SIZE>, BATCH_SIZE>& input,
		const CudaArray<Vector<T, OUTPUT_SIZE>, BATCH_SIZE>& expected_output,
		ActivationCuda::ActivationFunction activation,
		int maxThreadsPerBlock
	) {
		feed_forward(
			hidden_output_buff, 
			final_output_buff, 
			hidden_weights, 
			output_weights, 
			input, 
			activation, 
			maxThreadsPerBlock
		);
		find_errors(
			hidden_errors_buff, 
			output_errors_buff, 
			transposed_output_weights, 
			expected_output, 
			final_output_buff, 
			maxThreadsPerBlock
		);
		back_propagate(
			output_delta,
			hidden_delta,
			multiplication_buff_output,
			multiplication_buff_hidden,
			hidden_errors_buff,
			output_errors_buff,
			hidden_output_buff,
			final_output_buff,
			input, activation,
			maxThreadsPerBlock
		);
	}



	Matrix<T, HIDDEN_SIZE, INPUT_SIZE> hidden_weights;
	Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE> output_weights;
};
