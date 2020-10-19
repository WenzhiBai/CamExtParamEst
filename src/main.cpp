
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <cstring>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <ceres/ceres.h>

void print_help()
{
	std::cout << "Usage: ./main calibration_txt_path" << std::endl;
	std::cout << "Example: ./main ../calibration.txt" << std::endl;
}

Eigen::Vector3d quaternion2euler_zyx(const Eigen::Quaterniond & q)
{
    Eigen::Vector3d euler;
    euler(0) = atan2(2.0 * (q.y() * q.z() + q.w() * q.x()),
        1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y()));
    euler(1) = asin(-2.0 * (q.x() * q.z() - q.w() * q.y()));
    euler(2) = atan2(2.0 *(q.x() * q.y() + q.w() * q.z()),
        1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
    return euler;
}

struct HomogeneousCostFunctor 
{
   HomogeneousCostFunctor(const Eigen::Quaterniond & det_q_wheel,
   		const Eigen::Vector3d & det_pos_wheel, const Eigen::Vector3d & det_pos_camera)
	   :_det_q_wheel(det_q_wheel), _det_pos_wheel(det_pos_wheel), _det_pos_camera(det_pos_camera)
	{
	}

	template <typename T>
	bool operator()(const T* const q, const T* const t, 
		const T* const scale, T* residual) const 
	{
		Eigen::Quaternion<T> quat(q[3], q[0], q[1], q[2]);
		Eigen::Matrix<T, 3, 1> trans(t[0], t[1], t[2]);
		Eigen::Quaternion<T> det_q_wheel(T(_det_q_wheel.w()), T(_det_q_wheel.x()), T(_det_q_wheel.y()), T(_det_q_wheel.z()));
		Eigen::Matrix<T, 3, 1> det_pos_wheel(T(_det_pos_wheel(0)), T(_det_pos_wheel(1)), T(_det_pos_wheel(2)));
		Eigen::Matrix<T, 3, 1> det_pos_camera(T(_det_pos_camera(0)), T(_det_pos_camera(1)), T(_det_pos_camera(2)));
		Eigen::Matrix<T, 3, 1> res_error = det_q_wheel * trans 
			- trans + det_pos_wheel - scale[0] * (quat * det_pos_camera);
		for (size_t i = 0; i < 3; ++i)
		{
			residual[i] = res_error(i);
		}
		return true;
	}

	Eigen::Quaterniond _det_q_wheel;
	Eigen::Vector3d _det_pos_wheel;
	Eigen::Vector3d _det_pos_camera;
};

Eigen::Affine3d camera_calib(const std::vector<Eigen::Vector3d> & delta_pos_wheel_vec,
	const std::vector<Eigen::Quaterniond> & delta_q_wheel_vec,
	const std::vector<Eigen::Vector3d> & delta_pos_cam_vec,
	const std::vector<Eigen::Quaterniond> & delta_q_cam_vec,
	const Eigen::Vector3d & translation_wheel2camera)
{
	Eigen::Matrix<double, 3, Eigen::Dynamic> src;
    Eigen::Matrix<double, 3, Eigen::Dynamic> des;
    int cols = delta_pos_wheel_vec.size();
    src.resize(3, cols);
    des.resize(3, cols);
	for (size_t i = 0; i < cols; i++)
	{
		src.col(i) = delta_pos_cam_vec[i];

		Eigen::Vector3d Pwv = delta_q_wheel_vec[i] * translation_wheel2camera 
			- translation_wheel2camera + delta_pos_wheel_vec[i];
		des.col(i) = Pwv;
	}
	
	Eigen::Affine3d Tvc = Eigen::Affine3d(Eigen::umeyama(src, des, true));
    double scale = (Tvc.linear() * Tvc.rotation().transpose())(0, 0); 
	Eigen::Quaterniond q = Eigen::Quaterniond(Tvc.rotation());
	Eigen::Vector3d trans = translation_wheel2camera;
	Eigen::Matrix3d m;
    m << 1, 0, 0,
         0, 0, 1,
         0,-1, 0;
	Eigen::Vector3d roll_pitch_yaw = quaternion2euler_zyx(q * Eigen::Quaterniond(m.transpose()));
	std::swap(roll_pitch_yaw(0), roll_pitch_yaw(1));
	roll_pitch_yaw(1) *= -1.0;
	roll_pitch_yaw = 180.0 / M_PI * roll_pitch_yaw;
	std::cout << "----------------------umeyama----------------------" << std::endl;
	std::cout << "scale = " << std::to_string(scale) << std::endl;
	std::cout << "x = " << trans.x()
			<< "m, y = " << trans.y() 
			<< "m, z = " << trans.z() 
			<< "m" << std::endl;
	std::cout << "roll = " << roll_pitch_yaw.x()
			<< "deg, yaw = " << roll_pitch_yaw.z() 
			<< "deg, pitch = " << roll_pitch_yaw.y() 
			<< "deg" << std::endl;

	// Build the problem.
	ceres::Problem problem;
	ceres::LossFunction* loss_function = NULL; //new ceres::CauchyLoss(0.05);
	ceres::LocalParameterization* quaternion_local_parameterization =
		new ceres::EigenQuaternionParameterization;

	for (size_t i = 0; i != delta_q_wheel_vec.size(); ++i)
	{
		ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<HomogeneousCostFunctor, 3, 4, 3, 1>
			(new HomogeneousCostFunctor(delta_q_wheel_vec[i], delta_pos_wheel_vec[i], delta_pos_cam_vec[i]));
		problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), trans.data(), &scale);
		problem.SetParameterization(q.coeffs().data(), quaternion_local_parameterization);	
	}
	// problem.SetParameterBlockConstant(&scale);
	problem.SetParameterBlockConstant(trans.data());

	// Run the solver!
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;
	ceres::Solver::Summary summary;
	Solve(options, &problem, &summary);
	// std::cout << summary.BriefReport() << "\n";

	roll_pitch_yaw = quaternion2euler_zyx(q * Eigen::Quaterniond(m.transpose()));
	std::swap(roll_pitch_yaw(0), roll_pitch_yaw(1));
	roll_pitch_yaw(1) *= -1.0;
	roll_pitch_yaw = 180.0 / M_PI * roll_pitch_yaw;

	std::cout << "----------------------ceres-----------------------" << std::endl;
	std::cout << "scale = " << std::to_string(scale) << std::endl;
	std::cout << "x = " << trans.x()
			<< "m, y = " << trans.y() 
			<< "m, z = " << trans.z() 
			<< "m" << std::endl;
	std::cout << "roll = " << roll_pitch_yaw.x()
			<< "deg, yaw = " << roll_pitch_yaw.z() 
			<< "deg, pitch = " << roll_pitch_yaw.y() 
			<< "deg" << std::endl;

	return Eigen::Translation3d(trans) * q;
}

int main(int argc,char **argv)
{
	if (argc != 2)
	{
		print_help();
		return -1;
	}
	
	Eigen::Vector3d gt_ryp_cam2wheel;
	Eigen::Vector3d gt_xyz_cam2wheel;
	std::vector<Eigen::Vector3d> delta_pos_wheel_vec;
	std::vector<Eigen::Quaterniond> delta_q_wheel_vec;
	std::vector<Eigen::Vector3d> delta_pos_cam_vec;
	std::vector<Eigen::Quaterniond> delta_q_cam_vec;
	std::vector<Eigen::Vector3d> pos_wheel_vec;
	std::vector<Eigen::Quaterniond> q_wheel_vec;
	std::vector<Eigen::Vector3d> pos_cam_vec;
	std::vector<Eigen::Quaterniond> q_cam_vec;

	std::ifstream input(argv[1]);
    if (input) {
	    std::string line_buff;
        while (std::getline(input, line_buff)) {
            std::stringstream ss;
            ss.precision(14);
            ss << line_buff;
            std::string temp;
            std::vector<std::string> str_vec;
            while (ss >> temp) {
                str_vec.push_back(temp);
            }

			double from_timestamp;
			double to_timestamp;
			Eigen::Vector3d delta_pos_wheel;
			Eigen::Quaterniond delta_q_wheel;
			Eigen::Vector3d delta_pos_cam;
			Eigen::Quaterniond delta_q_cam;
			Eigen::Vector3d pos_wheel;
			Eigen::Quaterniond q_wheel;
			Eigen::Vector3d pos_cam;
			Eigen::Quaterniond q_cam;

            if (str_vec.size() == 30) {
				from_timestamp = std::stold(str_vec[0]);
				to_timestamp = std::stold(str_vec[1]);
				double delta_timestamp = to_timestamp - from_timestamp;
				if (delta_timestamp > 2 || delta_timestamp < 0.025)	// lowest speed: 0.66m/s; highest speed: 50km/h
				{
					std::cout << "delta_timestamp = " << delta_timestamp << std::endl;
					continue;
				}
				

                delta_pos_wheel = Eigen::Vector3d(std::stold(str_vec[2]), std::stold(str_vec[3]), std::stold(str_vec[4]));
                delta_pos_wheel_vec.emplace_back(delta_pos_wheel);
				delta_q_wheel = Eigen::Quaterniond(std::stold(str_vec[5]), std::stold(str_vec[6]), 
                                            std::stold(str_vec[7]), std::stold(str_vec[8]));
                delta_q_wheel_vec.emplace_back(delta_q_wheel);

				delta_pos_cam = Eigen::Vector3d(std::stold(str_vec[9]), std::stold(str_vec[10]), std::stold(str_vec[11]));
                delta_pos_cam_vec.emplace_back(delta_pos_cam);
				delta_q_cam = Eigen::Quaterniond(std::stold(str_vec[12]), std::stold(str_vec[13]), 
                                            std::stold(str_vec[14]), std::stold(str_vec[15]));
                delta_q_cam_vec.emplace_back(delta_q_cam);

				pos_wheel = Eigen::Vector3d(std::stold(str_vec[16]), std::stold(str_vec[17]), std::stold(str_vec[18]));
                pos_wheel_vec.emplace_back(pos_wheel);
				q_wheel = Eigen::Quaterniond(std::stold(str_vec[19]), std::stold(str_vec[20]), 
                                            std::stold(str_vec[21]), std::stold(str_vec[22]));
                q_wheel_vec.emplace_back(q_wheel);

				pos_cam = Eigen::Vector3d(std::stold(str_vec[23]), std::stold(str_vec[24]), std::stold(str_vec[25]));
                pos_cam_vec.emplace_back(pos_cam);
				q_cam = Eigen::Quaterniond(std::stold(str_vec[26]), std::stold(str_vec[27]), 
                                            std::stold(str_vec[28]), std::stold(str_vec[29]));
                q_cam_vec.emplace_back(q_cam);
            } else if (str_vec.size() == 19) {
				for (auto &str : str_vec) {
					std::string::size_type n;
					n = str.find(",");
					str = str.substr(0, n);
				}
				gt_ryp_cam2wheel = Eigen::Vector3d(std::stold(str_vec[3]), std::stold(str_vec[6]), std::stold(str_vec[9]));
				gt_xyz_cam2wheel = Eigen::Vector3d(std::stold(str_vec[12]), std::stold(str_vec[15]), std::stold(str_vec[18]));
				gt_xyz_cam2wheel = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()) * gt_xyz_cam2wheel;
			} else {
				std::cout << "data lost or header, and str_vec.size() = " << str_vec.size() << std::endl;
				for (size_t i = 0; i < str_vec.size(); i++)
				{
					std::cout << str_vec[i] << " ";
				}
				std::cout << std::endl;
				continue;
			}
        }
    } else {
		std::cout << "cannot open the file" << std::endl;
		return -1;
	}
    input.close();

	// process
	std::cout << "--------------------groundtruth-----------------------" << std::endl;
	std::cout << "x = " << gt_xyz_cam2wheel.x()
			<< "m, y = " << gt_xyz_cam2wheel.y() 
			<< "m, z = " << gt_xyz_cam2wheel.z() 
			<< "m" << std::endl;
	std::cout << "roll = " << gt_ryp_cam2wheel.x()
			<< "deg, yaw = " << gt_ryp_cam2wheel.y() 
			<< "deg, pitch = " << gt_ryp_cam2wheel.z() 
			<< "deg" << std::endl;

	std::cout << "------------------------------------------------------" << std::endl;
	std::cout << "----------------------all data------------------------" << std::endl;
	std::cout << "------------------------------------------------------" << std::endl;
	Eigen::Affine3d Twheel2camera = camera_calib(delta_pos_wheel_vec, delta_q_wheel_vec, 
		delta_pos_cam_vec, delta_q_cam_vec, gt_xyz_cam2wheel);

	std::cout << "--------------------------------------------------------" << std::endl;
	std::cout << "-----------------small rotation data--------------------" << std::endl;
	std::cout << "--------------------------------------------------------" << std::endl;
	std::vector<Eigen::Vector3d> small_delta_pos_wheel_vec;
	std::vector<Eigen::Quaterniond> small_delta_q_wheel_vec;
	std::vector<Eigen::Vector3d> small_delta_pos_cam_vec;
	std::vector<Eigen::Quaterniond> small_delta_q_cam_vec;
	for (size_t i = 0; i < delta_pos_wheel_vec.size(); i++)
	{
		if (std::fabs(delta_q_wheel_vec[i].angularDistance(Eigen::Quaterniond::Identity())) < 1 * M_PI / 180.0)
		{
			small_delta_pos_wheel_vec.emplace_back(delta_pos_wheel_vec[i]);
			small_delta_q_wheel_vec.emplace_back(delta_q_wheel_vec[i]);
			small_delta_pos_cam_vec.emplace_back(delta_pos_cam_vec[i]);
			small_delta_q_cam_vec.emplace_back(delta_q_cam_vec[i]);
		}
	}
	std::cout << "small ratio = " << 1.0 * small_delta_pos_wheel_vec.size() / delta_pos_wheel_vec.size() 
			<< ", small size = " << small_delta_pos_wheel_vec.size()
			<< ", total size = " << delta_pos_wheel_vec.size() << std::endl;
	Twheel2camera = camera_calib(small_delta_pos_wheel_vec, small_delta_q_wheel_vec, 
		small_delta_pos_cam_vec, small_delta_q_cam_vec, gt_xyz_cam2wheel);

	std::cout << "--------------------------------------------------------" << std::endl;
	std::cout << "-----------------large rotation data--------------------" << std::endl;
	std::cout << "--------------------------------------------------------" << std::endl;
	std::vector<Eigen::Vector3d> large_delta_pos_wheel_vec;
	std::vector<Eigen::Quaterniond> large_delta_q_wheel_vec;
	std::vector<Eigen::Vector3d> large_delta_pos_cam_vec;
	std::vector<Eigen::Quaterniond> large_delta_q_cam_vec;
	for (size_t i = 0; i < delta_pos_wheel_vec.size(); i++)
	{
		if (std::fabs(delta_q_wheel_vec[i].angularDistance(Eigen::Quaterniond::Identity())) > 4 * M_PI / 180.0)
		{
			large_delta_pos_wheel_vec.emplace_back(delta_pos_wheel_vec[i]);
			large_delta_q_wheel_vec.emplace_back(delta_q_wheel_vec[i]);
			large_delta_pos_cam_vec.emplace_back(delta_pos_cam_vec[i]);
			large_delta_q_cam_vec.emplace_back(delta_q_cam_vec[i]);
		}
	}
	std::cout << "large ratio = " << 1.0 * large_delta_pos_wheel_vec.size() / delta_pos_wheel_vec.size() 
			<< ", large size = " << large_delta_pos_wheel_vec.size()
			<< ", total size = " << delta_pos_wheel_vec.size() << std::endl;
	Twheel2camera = camera_calib(large_delta_pos_wheel_vec, large_delta_q_wheel_vec, 
		large_delta_pos_cam_vec, large_delta_q_cam_vec, gt_xyz_cam2wheel);

	return 0;
}

