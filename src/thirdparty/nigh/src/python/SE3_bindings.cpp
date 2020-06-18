#include <cstddef>
#include <assert.h>
#include <iostream>
#include <string>
#include <typeinfo>
#include <algorithm>
#include <utility>
#include <chrono>
#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <nigh/lp_space.hpp>
#include <nigh/scaled_space.hpp>
#include <nigh/so3_space.hpp>
#include <nigh/cartesian_space.hpp>
#include <nigh/kdtree_batch.hpp>

namespace nigh = unc::robotics::nigh;

using Space = nigh::CartesianSpace<
        nigh::ScaledSpace<nigh::SO3Space<double>>,
        nigh::L2Space<double, 3>>;
using State = std::tuple<Eigen::Quaterniond, Eigen::Vector3d>;

// State (points stored in tree) is the Cartesian product of SO(3) and R^3. Store node object
// which allows for retrieval of state using MyNodeKey function.
struct MyNode {
    std::string name_;
    State point_;

    MyNode(const std::string& name, const State pt)
        : name_(name)
        , point_(pt)
    {
    }
};

// Retrieval key function to retrieve underlying point from node.
struct MyNodeKey {
    const State& operator () (const MyNode& node) const {
        return node.point_;
    }
};

// custom class for weighted SE(3) which is initialized only by scale parameter
using NNTree = nigh::Nigh<MyNode, Space, MyNodeKey, nigh::Concurrent, nigh::KDTreeBatch<>>;
template<std::size_t T>
using MatrixXT = Eigen::Matrix<double, Eigen::Dynamic, T>; 
template<typename T>
using MatrixXY = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>; 

class SE3Tree {
    private:
    NNTree nigh_tree_;

    public:
    SE3Tree(double so3wt) : nigh_tree_([](double w){Space space(w, nigh::L2Space<double, 3>{}); return space;}(so3wt))
    {
    }
    void insert(MatrixXT<3> translations, MatrixXT<4> quaternions) {
        // number of translations and rotations must be equal
        assert (translations.rows() == quaternions.rows());
        for (uint i = 0; i < translations.rows(); i++) {
            Eigen::Quaterniond quat(quaternions.row(i).transpose().normalized());
            Eigen::Vector3d t = translations.row(i).transpose();
            State pt{quat, t};
            nigh_tree_.insert(MyNode{std::to_string(i), pt});
        }
    }
    std::pair<MatrixXY<double>, MatrixXY<int>> nearest(MatrixXT<3> translations, MatrixXT<4> quaternions, std::size_t k, uint n_jobs) {
        // number of translations and rotations must be equal
        assert (translations.rows() == quaternions.rows());
        // datastructure for indices and distances
        MatrixXY<int> id_mat(translations.rows(), k);
        MatrixXY<double> dist_mat(translations.rows(), k);

        omp_set_num_threads(n_jobs);

        #pragma omp parallel for
        for (uint i = 0; i < translations.rows(); i++) {
            std::vector<std::pair<MyNode, double>> nbh;
            Eigen::Quaterniond quat(quaternions.row(i).transpose().normalized());
            Eigen::Vector3d t = translations.row(i).transpose();
            State pt{quat, t};
            nigh_tree_.nearest(nbh, pt, k);
            for (uint j = 0; j < nbh.size(); j++) {
                // return index of retireved tree element
                id_mat(i, j) = std::stoi(std::get<0>(nbh[j]).name_);
                dist_mat(i, j) = std::get<1>(nbh[j]);
            }
        }
        std::pair<MatrixXY<double>, MatrixXY<int>> out(dist_mat, id_mat);
        return out;
    }
};


namespace py = pybind11;

PYBIND11_MODULE(Nigh, m) {
    py::class_<SE3Tree>(m, "SE3Tree")
        .def(py::init<double>())
        .def("insert", &SE3Tree::insert)
        .def("nearest", &SE3Tree::nearest);
}
