/*
 * kinematics.hpp
 *
 * Created on: Sep. 10 2013
 * Author: Amir Yazdanbakhsh <yazdanbakhsh@wisc.edu>
 */

 #ifndef __KINEMATICS_HPP__
 #define __KINEMATICS_HPP__

 void forwardk2j(float theta1, float theta2, float* x, float* y);
 void inversek2j(float x, float y, float* theta1, float* theta2);

 #endif
