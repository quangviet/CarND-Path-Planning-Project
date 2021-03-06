#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

#define MAX_VELOCITY 21.5
#define INC_VELOCITY 0.17

#define REF_DISTANCE 30
#define STATE_KEEP_LANE 0
#define STATE_SWITCH_LEFT 1
#define STATE_SWITCH_RIGHT 2
#define STATE_FOLLOW 3

#define FRONT_SAFE_DISTANCE 25
#define REAR_SAFE_DISTANCE 8

#define OUTPUT_POINTS 50
#define MUST_KEEP_POINTS 5

using namespace std;

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

double distance(double x1, double y1, double x2, double y2)
{
  return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y)
{

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

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{

  int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

  double map_x = maps_x[closestWaypoint];
  double map_y = maps_y[closestWaypoint];

  double heading = atan2( (map_y-y),(map_x-x) );

  double angle = abs(theta-heading);

  if(angle > pi()/4)
  {
    closestWaypoint++;
  }

  return closestWaypoint;

}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{
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
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y)
{
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

  double ref_velocity = 0;       
  double ref_lane = 1;

  h.onMessage([&ref_velocity, &ref_lane, &map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
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
          
          // Main car's localization Data
            double car_x = j[1]["x"];
            double car_y = j[1]["y"];
            double car_s = j[1]["s"];
            double car_d = j[1]["d"];
            double car_yaw = j[1]["yaw"];
            double car_speed = j[1]["speed"];

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

            double theta = deg2rad(car_yaw);

            vector<double> spline_x;
            vector<double> spline_y;

            double ref_s = car_s;

            /*
              Check if the previous path still valid 
            */
            bool last_path_ok = true;

            vector<double> path_s;
            vector<double> path_d;
            for (int i = 0; i < previous_path_x.size(); i++) {
                auto sd = getFrenet(previous_path_x[i], previous_path_y[i], theta, map_waypoints_x, map_waypoints_y);
                path_s.push_back(sd[0]);
                path_d.push_back(sd[1]);
            }

            for (int i = 0; i < sensor_fusion.size(); i++) {
              double obj_vx = sensor_fusion[i][3];
              double obj_vy = sensor_fusion[i][4];
              double obj_v = sqrt(obj_vx * obj_vx + obj_vy * obj_vy);
              double obj_s = sensor_fusion[i][5];
              double obj_d = sensor_fusion[i][6];

              for (int j = 0; j < previous_path_x.size(); j++) {
                double pred_s = obj_s + obj_v * (j+1) * 0.02;
                if ((abs(path_s[j] - pred_s) < 4) && (abs(path_d[j] - obj_d) < 2.3) && (obj_s > car_s - 2)) {
                  auto xy = getXY(pred_s, obj_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
                  if (distance(xy[0], xy[1], previous_path_x[j], previous_path_y[j]) < 4) {
                    last_path_ok = false;
                    break;
                  }
                }
              }
            }

            int reused_points = previous_path_x.size();

            /*
              We change the reference speed and lane accordingly in case last path was not reusable
            */
            if (!last_path_ok) {
              reused_points = MUST_KEEP_POINTS;

              end_path_s = path_s[MUST_KEEP_POINTS - 1];
              end_path_d = path_d[MUST_KEEP_POINTS - 1];
              
              if (end_path_d < 4) {
                ref_lane = 0;
              } else if (end_path_d < 8) {
                ref_lane = 1;
              } else {
                ref_lane = 2;
              }

              double car_x = previous_path_x[reused_points - 1];
              double car_y = previous_path_y[reused_points - 1];

              double car_lx = previous_path_x[reused_points - 2];
              double car_ly = previous_path_y[reused_points - 2];              

              ref_velocity = distance(car_x, car_y, car_lx, car_ly) * 50;
            }

            /*
              We reused all the previously generated points and shift/rotate these poitns to local coordinates
            */
            if (reused_points > 0) {
              car_x = previous_path_x[reused_points - 1];
              car_y = previous_path_y[reused_points - 1];

              double car_lx = previous_path_x[reused_points - 2];
              double car_ly = previous_path_y[reused_points - 2];

              theta = atan2(car_y - car_ly, car_x - car_lx);
              ref_s = end_path_s;

              for (int i = 0; i < reused_points;i++) {
                double px = previous_path_x[i];
                double py = previous_path_y[i];

                next_x_vals.push_back(px);
                next_y_vals.push_back(py);

                globalToLocal(px, py, theta, car_x, car_y);
                spline_x.push_back(px);
                spline_y.push_back(py);
              } 
            } else {
              spline_x.push_back(0);
              spline_y.push_back(0);
            }

            /*
              Check the feasibilities of 3 possible action: Keep current lane, switch left or switch right
            */

            bool keep_lane_ok = true;
            bool switch_left_ok = true;            
            bool switch_right_ok = true;

            for (int i = 0; i < sensor_fusion.size(); i++) {
              double obj_vx = sensor_fusion[i][3];
              double obj_vy = sensor_fusion[i][4];
              double obj_v = sqrt(obj_vx * obj_vx + obj_vy * obj_vy);
              double obj_s = double(sensor_fusion[i][5]) + obj_v * reused_points * 0.02;
              double obj_d = sensor_fusion[i][6];

              if ((abs(obj_d - (ref_lane * 4 + 2)) < 2.5) && (obj_s > ref_s) && (obj_s < ref_s + FRONT_SAFE_DISTANCE)) {
                keep_lane_ok = false;
              }

              if ((abs(obj_d - (ref_lane * 4 - 2)) < 2.5) && (obj_s > ref_s - REAR_SAFE_DISTANCE) && (obj_s < ref_s + FRONT_SAFE_DISTANCE)) {
                switch_left_ok = false;
              }

              if ((abs(obj_d - (ref_lane * 4 + 6)) < 2.5) && (obj_s > ref_s - REAR_SAFE_DISTANCE) && (obj_s < ref_s + FRONT_SAFE_DISTANCE)) {
                switch_right_ok = false;
              }
            }

            /*
              My simple state machine
            */

            int next_action = STATE_KEEP_LANE;
            if (!keep_lane_ok) {
              next_action = STATE_FOLLOW;
              if ((ref_lane > 0) && switch_left_ok) {
                next_action = STATE_SWITCH_LEFT;
                ref_lane = ref_lane - 1;
              } 

              if ((next_action == 3) && (ref_lane < 2) && switch_right_ok){
                next_action = STATE_SWITCH_RIGHT;
                ref_lane = ref_lane + 1;
              } 
            }

            if (next_action == STATE_FOLLOW) {
              ref_velocity -= INC_VELOCITY;
              if (ref_velocity < 0) {
                ref_velocity = 0;
              }
            } else if (next_action == STATE_KEEP_LANE) {
              if (ref_velocity < MAX_VELOCITY) {
                ref_velocity += INC_VELOCITY;
              }
              if (ref_velocity > MAX_VELOCITY) {
                ref_velocity = MAX_VELOCITY;
              }
            } else {
              if (ref_velocity > MAX_VELOCITY - 10) {
                ref_velocity -= INC_VELOCITY;
              }
            }

            /*
              Create some extra points in the future path and smooth the outputs. 
              Then I reversed back to the global coordinate and return the path.
            */

            for (int i = 1; i < 4; i++) {
              auto xy = getXY(ref_s + REF_DISTANCE * i, ref_lane * 4 + 2, map_waypoints_s, map_waypoints_x, map_waypoints_y);
              double px = xy[0];
              double py = xy[1];

              globalToLocal(px, py, theta, car_x, car_y);
              spline_x.push_back(px);
              spline_y.push_back(py);
            }

            tk::spline ref_spline;
            ref_spline.set_points(spline_x, spline_y);

            double dist = distance(0, 0, REF_DISTANCE, ref_spline(REF_DISTANCE));
            double delta_x = REF_DISTANCE * ref_velocity / (dist * 50);

            double current_x = 0;

            while (next_x_vals.size() < OUTPUT_POINTS) {
              current_x +=  delta_x;
              double lx = current_x;
              double ly = ref_spline(lx);
              localToGlobal(lx, ly, theta, car_x, car_y);

              next_x_vals.push_back(lx);
              next_y_vals.push_back(ly);
            }

            msgJson["next_x"] = next_x_vals;
            msgJson["next_y"] = next_y_vals;

            auto msg = "42[\"control\","+ msgJson.dump()+"]";

            //this_thread::sleep_for(chrono::milliseconds(1000));
            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
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
