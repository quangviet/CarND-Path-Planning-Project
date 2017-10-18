#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "Eigen-3.3/Eigen/Dense"
#include "json.hpp"
#include "spline.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Number of points to be kept from previous path
#define KEEP_POINTS 5

// Number of points in the new path
#define PREDICT_POINTS 50

// Duration of each trajectory
#define PREDICT_TIME 2

// Maximum veolocity of the vehicle in meter per second
#define MAX_VELOCITY 22.2

// For visual debug
#define ZOOM_RATIO 8
#define VISUAL_RANGE 800
#define VISUAL_CENTER 400
#define VISUAL_DEBUG

using namespace cv;
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2) {
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y) {
	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;
}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y) {
	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2( (map_y-y),(map_x-x) );

	double angle = abs(theta-heading);

	if(angle > pi()/4)
	{
		closestWaypoint++;
	}

  if (closestWaypoint >= maps_x.size()) {
    closestWaypoint = 0;
  }

	return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y) {
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};
}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y) {
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};
}

vector<double> JMT(vector< double> start, vector <double> end, double T) {
  MatrixXd A = MatrixXd(3, 3);
  A << T*T*T, T*T*T*T, T*T*T*T*T,
          3*T*T, 4*T*T*T,5*T*T*T*T,
          6*T, 12*T*T, 20*T*T*T;
    
  MatrixXd B = MatrixXd(3,1);     
  B << end[0]-(start[0]+start[1]*T+.5*start[2]*T*T),
          end[1]-(start[1]+start[2]*T),
          end[2]-start[2];
          
  MatrixXd Ai = A.inverse();
  
  MatrixXd C = Ai*B;
  
  vector <double> result = {start[0], start[1], .5*start[2]};
  for(int i = 0; i < C.size(); i++)
  {
      result.push_back(C.data()[i]);
  }
  
  return result;  
}

void globalToLocal(double &x, double &y, double theta, double dx, double dy) {
  double tx = x - dx;
  double ty = y - dy;                 
  x = tx * cos(-theta) - ty * sin(-theta);
  y = ty * cos(-theta) + tx * sin(-theta);         
}

void localToGlobal(double &x, double &y, double theta, double dx, double dy) {
  double tx = x * cos(theta) - y * sin(theta);
  double ty = y * cos(theta) + x * sin(theta);         
  x = dx + tx;
  y = dy + ty;                  
}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }
  
  #ifdef VISUAL_DEBUG
  namedWindow("Trajectory", WINDOW_AUTOSIZE);
  #endif

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object

          typedef std::chrono::high_resolution_clock clock;
          std::chrono::time_point<clock> start = clock::now();           
          
        	// Main car's localization Data
        	double car_x = j[1]["x"];
        	double car_y = j[1]["y"];
        	double car_s = j[1]["s"];
        	double car_d = j[1]["d"];
        	double car_yaw = j[1]["yaw"];
        	double car_speed = j[1]["speed"];

          // Convert yaw to radians, speed to meter per second
          car_yaw = deg2rad(car_yaw);
          car_speed *= 0.44704;
        	
        	//printf("x = %7.2f, y = %7.2f, v = %5.2f", car_x, car_y, car_speed);

        	// Previous path data given to the Planner
        	auto previous_path_x = j[1]["previous_path_x"];
        	auto previous_path_y = j[1]["previous_path_y"];
        	// Previous path's end s and d values 
        	double end_path_s = j[1]["end_path_s"];
        	double end_path_d = j[1]["end_path_d"];

        	// Sensor Fusion Data, a list of all other cars on the same side of the road.
        	auto sensor_fusion = j[1]["sensor_fusion"];

        	json msgJson;

        	vector<double> next_x_vals;
        	vector<double> next_y_vals;

          /*
            I calculate the base parameter of the car. If this is the first time, the previous_path.size() will be zero.
            In this case, base car parameter is based on the current car information.

            If previous path existed, we keep a certain number of points from the previous path (KEEP_POINTS), and base
            parameter will based on the last points among the points we kept.

            I calculated the velocity and acceleration in s and d (vs, vd, as, ad) as well.
            I also added some point from the previous path to the spline reference points.
          */
          double base_car_x = car_x;
          double base_car_y = car_y;

          double base_car_s = car_s;
          double base_car_d = car_d;

          double base_car_vs = 0;
          double base_car_vd = 0;
          double base_car_as = 0;
          double base_car_ad = 0;
          
          double base_car_theta = car_yaw;

          vector<double> spline_x;
          vector<double> spline_y;

          if (previous_path_x.size() > KEEP_POINTS) {
            base_car_x = previous_path_x[KEEP_POINTS];
            base_car_y = previous_path_y[KEEP_POINTS];

            for (int i = 0; i < KEEP_POINTS; i++) {
              next_x_vals.push_back(double(previous_path_x[i]));
              next_y_vals.push_back(double(previous_path_y[i]));
            }

            double last_car_x = previous_path_x[KEEP_POINTS-1];
            double last_car_y = previous_path_y[KEEP_POINTS-1];

            double last_last_car_x = previous_path_x[KEEP_POINTS-2];
            double last_last_car_y = previous_path_y[KEEP_POINTS-2];

            base_car_theta = atan2(base_car_y - last_car_y, base_car_x - last_car_x);
            auto sd = getFrenet(base_car_x, base_car_y, base_car_theta, map_waypoints_x, map_waypoints_y);

            base_car_s = sd[0];
            base_car_d = sd[1];

            double velocity = distance(base_car_x, base_car_y, last_car_x, last_car_y) * 50;

            int next_wp = NextWaypoint(base_car_x, base_car_y, base_car_theta, map_waypoints_x, map_waypoints_y);
            int prev_wp = next_wp - 1;
            if (prev_wp < 0) {
              prev_wp = map_waypoints_x.size() - 1;
            }

            double waypoint_theta = atan2(map_waypoints_y[next_wp] - map_waypoints_y[prev_wp], map_waypoints_x[next_wp] - map_waypoints_x[prev_wp]);
            double diff_theta = waypoint_theta - base_car_theta;

            //printf(" next_wp = %d prev_wp = %d\n", next_wp, prev_wp);
            //printf(", diff_theta = %9.3f waypoint_theta = %9.3f base_car_theta = %9.3f", diff_theta, waypoint_theta, base_car_theta);

            base_car_vs = velocity * cos(diff_theta);
            base_car_vd = velocity * sin(diff_theta);

            double vx = (base_car_x - last_car_x) * 50;
            double vy = (base_car_y - last_car_y) * 50;

            double last_vx = (last_car_x - last_last_car_x) * 50;
            double last_vy = (last_car_y - last_last_car_y) * 50;

            double ax = vx - last_vx;
            double ay = vy - last_vy;

            double accel = distance(vx, vy, last_vx, last_vy);
            double accel_theta = atan2(ay, ax);
            double diff_accel_theta = waypoint_theta - accel_theta;

            base_car_as = accel * cos(diff_accel_theta);
            base_car_ad = accel * sin(diff_accel_theta);     

            double start_x = previous_path_x[0];
            double start_y = previous_path_y[0];

            globalToLocal(start_x, start_y, base_car_theta, base_car_x, base_car_y);

            spline_x.push_back(start_x);
            spline_x.push_back(0);

            spline_y.push_back(start_y);
            spline_y.push_back(0);
          } else {
            spline_x.push_back(0);
            spline_y.push_back(0);
          }

          //printf(", base_car_as = %9.3f, base_car_ad = %9.3f ", base_car_as, base_car_ad);

          #ifdef VISUAL_DEBUG
          Mat image = Mat::zeros(VISUAL_RANGE, VISUAL_RANGE, CV_8UC3);

          int wp = ClosestWaypoint(car_x, car_y, map_waypoints_x, map_waypoints_y) - 3;
          if (wp < 0) {
            wp = wp + map_waypoints_x.size();
          }

          for (int i = 0; i < 8; i++) {
            int next_wp = wp + 1;
            if (next_wp >= map_waypoints_x.size()) {
              next_wp = 0;
            }

            for (int j = 0; j < 4; j++) {
              double fx = map_waypoints_x[wp] + map_waypoints_dx[wp] * j * 4;
              double fy = map_waypoints_y[wp] + map_waypoints_dy[wp] * j * 4;
              double tx = map_waypoints_x[next_wp] + map_waypoints_dx[next_wp] * j * 4;
              double ty = map_waypoints_y[next_wp] + map_waypoints_dy[next_wp] * j * 4;

              cv::line(image, 
                       Point((fx - car_x) * ZOOM_RATIO + VISUAL_CENTER, (car_y - fy) * ZOOM_RATIO + VISUAL_CENTER), 
                       Point((tx - car_x) * ZOOM_RATIO + VISUAL_CENTER, (car_y - ty) * ZOOM_RATIO + VISUAL_CENTER), 
                       Scalar(200, 200, 200));            
            }

            wp = next_wp;
          }
          #endif

          double best_v;
          double best_s;
          double best_a;
          double best_cost = 1e+15;
          
          double best_d;
          double best_d_time;

          /*
            Because repeately calculating the JMT will comsume quite alot of computing power, I precalculated the JMT and store
            them in this vector. Later all I have to do is lookup the value.
          */

          vector<vector<double>> d_jmt;

          for (double target_d = 2; target_d <= 10; target_d += 4) {
            for (double d_time = 0.1; d_time <= PREDICT_TIME; d_time += 0.2) {
              d_jmt.push_back(JMT({base_car_d, base_car_vd, base_car_ad}, {target_d, 0, 0}, d_time));
            }
          }

          /*
            I also precalculated the approximately position of other cars in each point of time. 
            This speed up thing a little bit.

            Here I assume that all other cars have constant velocity in frenet coordinates, which means
            they are moving along the road in constant rate (velocity).
          */

          double obj_pred_s[20][10];
          double obj_pred_d[20]; 

          int fusion_size = sensor_fusion.size();
          for (int i = 0; i < fusion_size; i++) {
            double obj_vx = sensor_fusion[i][3];
            double obj_vy = sensor_fusion[i][4];
            double obj_s = sensor_fusion[i][5];
            double obj_v = sqrt(obj_vx * obj_vx + obj_vy * obj_vy);
            obj_pred_d[i] = sensor_fusion[i][6];

            int t_count = 0;
            for (double t = 0.1; t <= PREDICT_TIME; t += 0.2) {
              obj_pred_s[i][t_count] = obj_s + obj_v * (t + 0.02 * next_x_vals.size());
              t_count++;
            }
          }

          /*
            Here I generate a number of trajectories using quintic polynomials.
            JMT was used to find the solution of the polynomials.

            For each trajactory generated, I calculated the cost and find the the trajectory with the smallest 
            cost.
          */

          for (double target_v = 7.5; target_v <= (MAX_VELOCITY+2); target_v += 2) {
            for (double target_s = 10; target_s <= (MAX_VELOCITY+2) * PREDICT_TIME; target_s += 4) {
              for (double target_a = -4; target_a <= 4; target_a += 2) {
                auto coeff = JMT({0, base_car_vs, base_car_as}, {target_s, target_v, target_a}, PREDICT_TIME);

                int d_jmt_count = 0;
                for (double target_d = 2; target_d <= 10; target_d += 4) {
                  for (double d_time = 0.1; d_time <= PREDICT_TIME; d_time += 0.2) {
                    vector<double> d_coeff = d_jmt[d_jmt_count];
                    d_jmt_count++;
              
                    double cost = (MAX_VELOCITY * PREDICT_TIME - target_s) * 10000000 + abs(target_d - base_car_d) * 1000; 

                    #ifdef VISUAL_DEBUG
                    vector<double> list_x;
                    vector<double> list_y;
                    double last_x = (base_car_x - car_x) * ZOOM_RATIO + VISUAL_CENTER;
                    double last_y = (car_y - base_car_y) * ZOOM_RATIO + VISUAL_CENTER;
                    list_x.push_back(last_x);
                    list_y.push_back(last_y);
                    #endif

                    int t_count = 0;
                    for (double t = 0.1; t <= PREDICT_TIME; t += 0.2) {
                      double p_s = base_car_s 
                                  + coeff[0] 
                                  + coeff[1] * t 
                                  + coeff[2] * t * t 
                                  + coeff[3] * t * t * t
                                  + coeff[4] * t * t * t * t
                                  + coeff[5] * t * t * t * t * t;

                      double p_d = target_d;
                      double vt_d = 0;
                      double at_d = 0;
                      double jt_d = 0;
                                  
                      if (t <= d_time) {
                        p_d = d_coeff[0] 
                            + d_coeff[1] * t 
                            + d_coeff[2] * t * t 
                            + d_coeff[3] * t * t * t
                            + d_coeff[4] * t * t * t * t
                            + d_coeff[5] * t * t * t * t * t;

                        vt_d = d_coeff[1] 
                             + 2 * d_coeff[2] * t 
                             + 3 * d_coeff[3] * t * t 
                             + 4 * d_coeff[4] * t * t * t
                             + 5 * d_coeff[5] * t * t * t * t;
                   
                        at_d = 2  * d_coeff[2] 
                             + 6  * d_coeff[3] * t 
                             + 12 * d_coeff[4] * t * t
                             + 20 * d_coeff[5] * t * t * t;
                           
                        jt_d = 6  * d_coeff[3] 
                             + 24 * d_coeff[4] * t
                             + 60 * d_coeff[5] * t * t;
                      }

                      #ifdef VISUAL_DEBUG
                      auto pxy = getXY(p_s, p_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
                      double draw_x = (pxy[0] - car_x) * ZOOM_RATIO + VISUAL_CENTER;
                      double draw_y = (car_y - pxy[1]) * ZOOM_RATIO + VISUAL_CENTER;
                      list_x.push_back(draw_x);
                      list_y.push_back(draw_y);
                      #endif
                    
                      double vt_s = coeff[1] 
                                  + 2 * coeff[2] * t 
                                  + 3 * coeff[3] * t * t 
                                  + 4 * coeff[4] * t * t * t
                                  + 5 * coeff[5] * t * t * t * t;
                         
                      double at_s = 2  * coeff[2] 
                                  + 6  * coeff[3] * t 
                                  + 12 * coeff[4] * t * t
                                  + 20 * coeff[5] * t * t * t;
                         
                      double jt_s = 6  * coeff[3] 
                                  + 24 * coeff[4] * t
                                  + 60 * coeff[5] * t * t;
                                  
                                  
                      double vt = sqrt(vt_s * vt_s + vt_d * vt_d);          

                      if (vt_s < 0){
                        cost = 1e+15;
                        break;
                      }

                      if (t < 0.2) {
                        if (abs(at_s) > 10) {
                          cost = 1e+15;
                          break;
                        }
                        
                        if (abs(at_d) > 10) {
                          cost = 1e+15;
                          break;
                        }           

                        if (vt > MAX_VELOCITY) {
                          cost = 1e+15;
                          break;
                        } 


                        if ((at_s*at_s + at_d * at_d) > 100) {
                          cost = 1e+15;
                          break;
                        }
             

                        if ((jt_s*jt_s + jt_d*jt_d) > 2500) {
                          cost = 1e+15;
                          break;
                        }

                      } else {                      
                        if (abs(vt) > MAX_VELOCITY + 3) {
                          cost = 1e+15;
                          break;
                        }
                        if (abs(at_s) > 50) {
                          cost = 1e+15;
                          break;
                        }
                        
                        if (abs(at_d) > 50) {
                          cost = 1e+15;
                          break;
                        }           

                        if ((at_s*at_s + at_d * at_d) > 1000) {
                          cost = 1e+15;
                          break;
                        }
             
                        if ((jt_s*jt_s + jt_d*jt_d) > 20000) {
                          cost = 1e+15;
                          break;
                        }

                      }

                      for (int i = 0; i < fusion_size; i++) {
                        double pred_s = obj_pred_s[i][t_count];
                        double pred_d = obj_pred_d[i];
                        if ((abs(p_s - pred_s) < 10) && (abs(p_d - pred_d) < 3)) {
                          cost = 1e+15;
                          break;
                        } else if (abs(p_d - pred_d) < 3) {
                          cost += 80000/abs(p_s - pred_s);
                        }
                      }
                      t_count++;
                     
                      double lane_diff = abs(target_d - p_d);                          

                      cost += (vt - MAX_VELOCITY) * (vt - MAX_VELOCITY) + lane_diff * lane_diff * 10000;
                      #ifdef VISUAL_DEBUG
                      if (cost >= 1e+15) {
                        break;
                      }
                      #else
                      if (cost >= best_cost) {
                        break;
                      }
                      #endif
                    }

                    if (cost < best_cost) {
                      best_cost = cost;
                      best_a = target_a;
                      best_v = target_v;
                      best_s = target_s;
                      
                      best_d = target_d;
                      best_d_time = d_time;
                    }                        

                    #ifdef VISUAL_DEBUG
                    if (cost < 1e+15) {
                      for (int i = 0; i < list_x.size() - 1; i++) {
                        cv::line(image, Point(list_x[i], list_y[i]), Point(list_x[i+1], list_y[i+1]), Scalar(0, 255, 0));
                      }
                    }
                    #endif
                  }
                }
                
              }
            }
          }

          /*
            Based on the trajectory with the optimized cost, I generated some reference points and added them to
            the spline. 

            Due to the inconsistencies in the conversion between the frenet frame and cartesian coordinates, I
            had to ignore some point which x < last x
          */

          double last_sp_x = 0;

          auto coeff = JMT({0, base_car_vs, base_car_as}, {best_s, best_v, best_a}, PREDICT_TIME);
          auto d_coeff = JMT({car_d, base_car_vd, base_car_ad}, {best_d, 0, 0}, best_d_time);

          for (double t = 0.5; t <= PREDICT_TIME; t += 0.5) {
            double car_s = base_car_s 
                           + coeff[0]
                           + coeff[1] * t 
                           + coeff[2] * t * t
                           + coeff[3] * t * t * t 
                           + coeff[4] * t * t * t * t
                           + coeff[5] * t * t * t * t * t;
                           
            double car_d = best_d;
            if (t <= best_d_time) {
              car_d = d_coeff[0]
                    + d_coeff[1] * t 
                    + d_coeff[2] * t * t
                    + d_coeff[3] * t * t * t 
                    + d_coeff[4] * t * t * t * t
                    + d_coeff[5] * t * t * t * t * t;
            }       

            auto xy = getXY(car_s, car_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
            double lx = xy[0];
            double ly = xy[1];

            globalToLocal(lx, ly, base_car_theta, base_car_x, base_car_y);
            if (lx > last_sp_x) {
              last_sp_x = lx;
              spline_x.push_back(lx);
              spline_y.push_back(ly);
            }
          }

          /*
            We now have a spline of the optimized trajactory. All we have to do is generate all the points based on
            the spline and added them to the next_x_vals and next_y_vals.
          */

          tk::spline traj_spline;
          traj_spline.set_points(spline_x, spline_y);

          for (double t = 0; t < PREDICT_TIME; t += 0.02) {
            double lx = coeff[0]
                      + coeff[1] * t 
                      + coeff[2] * t * t
                      + coeff[3] * t * t * t 
                      + coeff[4] * t * t * t * t
                      + coeff[5] * t * t * t * t * t;              

            double ly = traj_spline(lx);
            localToGlobal(lx, ly, base_car_theta, base_car_x, base_car_y);

            if (next_x_vals.size() < PREDICT_POINTS) {
              next_x_vals.push_back(lx);
              next_y_vals.push_back(ly);              
            }
          }

        	// TODO: define a path made up of (x,y) points that the car will visit sequentially every .02 seconds
        	msgJson["next_x"] = next_x_vals;
        	msgJson["next_y"] = next_y_vals;

        	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start).count();            
          //printf(" elapsed = %5.2fms\n", elapsed);

        	//this_thread::sleep_for(chrono::milliseconds(1000));
        	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);

          #ifdef VISUAL_DEBUG
          for (int i = 0; i < sensor_fusion.size(); i++) {
            double obj_x = sensor_fusion[i][1];
            double obj_y = sensor_fusion[i][2];
            double obj_vx = sensor_fusion[i][3];
            double obj_vy = sensor_fusion[i][4];
            
            circle(image, Point((obj_x - car_x) * ZOOM_RATIO + VISUAL_CENTER, (car_y - obj_y) * ZOOM_RATIO + VISUAL_CENTER), 1.5 * ZOOM_RATIO, Scalar(0, 0, 255));
            arrowedLine(
                image, 
                Point((obj_x - car_x) * ZOOM_RATIO + VISUAL_CENTER, (car_y - obj_y) * ZOOM_RATIO + VISUAL_CENTER),
                Point((obj_x - car_x + obj_vx * 0.5) * ZOOM_RATIO + VISUAL_CENTER, (car_y - obj_y - obj_vy * 0.5) * ZOOM_RATIO + VISUAL_CENTER),
                Scalar(0, 0, 255),
                1.5, 8, 0, 0.05
              );

            circle(
              image, 
              Point((obj_x - car_x + obj_vx * next_x_vals.size() * 0.02) * ZOOM_RATIO + VISUAL_CENTER, (car_y - obj_y - obj_vy * next_x_vals.size() * 0.02) * ZOOM_RATIO + VISUAL_CENTER),
              1.5 * ZOOM_RATIO, Scalar(0, 0, 100));            
          }

          double sx = VISUAL_CENTER;
          double sy = VISUAL_CENTER; 
          for (int i = 0; i < next_x_vals.size(); i++) {
            double dx = (next_x_vals[i] - car_x) * ZOOM_RATIO + VISUAL_CENTER;
            double dy = (car_y - next_y_vals[i]) * ZOOM_RATIO + VISUAL_CENTER;
            cv::line(image, Point(sx, sy), Point(dx, dy), Scalar(255, 255, 255));
            sx = dx;
            sy = dy;
          }

          circle(image, Point(VISUAL_CENTER, VISUAL_CENTER), 1.5 * ZOOM_RATIO, Scalar(255, 255, 255), -1);

          imshow("Trajectory", image);
          waitKey(1);
          #endif
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
