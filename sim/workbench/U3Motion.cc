#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/JointPosition.hh>
#include <gz/sim/components/JointVelocity.hh>
#include <gz/sim/components/JointVelocityCmd.hh>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/stringmsg.pb.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <vector>

class U3Motion : public gz::sim::System, public gz::sim::ISystemConfigure,
    public gz::sim::ISystemPreUpdate, public gz::sim::ISystemPostUpdate {
  using Entity = gz::sim::Entity;
  Entity rotor=gz::sim::kNullEntity, joint=gz::sim::kNullEntity, camera=gz::sim::kNullEntity;
  gz::transport::Node node;
  gz::transport::Node::Publisher publisher;
  std::vector<double> times, speeds;
  double baseX=0, baseY=0, baseZ=0, yawAmplitude=0, pitch=0, period=8, travel=0;
  double yawRelative=0, commandOmega=0;
  bool external=false, safetyFault=false;
  double lastTime=0, maxSpeed=0, maxAcceleration=0, maxRate=0, maxAngularAcceleration=0;
  double hubY=0, hubZ=0, axisY=1, axisZ=0, minStandOff=0;
  double gimbalYaw=0, gimbalPitch=0, yawRate=0, pitchRate=0;
  gz::math::Vector3d position, velocity;
  struct Command {
    int sequence=-1;
    gz::math::Vector3d position;
    double yaw=0,pitch=0,validSim=0,validWall=0;
  };
  Command pending, active;
  std::mutex commandMutex;
  double appliedAt=0,appliedWall=0;
  int seenSequence=-1, expiredSequence=-1;
  static double wallSeconds(){
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  static constexpr double rad=3.14159265358979323846/180;

  double omegaAt(double t) const {
    auto upper=std::upper_bound(times.begin(),times.end(),t);
    if (upper==times.begin()) return speeds.front();
    if (upper==times.end()) return speeds.back();
    const auto j=upper-times.begin();
    return speeds[j-1]+(speeds[j]-speeds[j-1])*(t-times[j-1])/(times[j]-times[j-1]);
  }

  void command(const gz::msgs::StringMsg &message) {
    Command candidate;
    double x,y,z;
    std::istringstream input(message.data());
    if(!(input>>candidate.sequence>>x>>y>>z>>candidate.yaw>>candidate.pitch>>candidate.validSim>>candidate.validWall))return;
    if(!std::isfinite(x+y+z+candidate.yaw+candidate.pitch+candidate.validSim+candidate.validWall))return;
    candidate.position.Set(x,y,z);
    std::lock_guard<std::mutex> lock(commandMutex);
    if(candidate.sequence>pending.sequence)pending=candidate;
  }

  static gz::math::Vector3d limit(gz::math::Vector3d vector,double bound) {
    if(vector.Length()>bound)vector=vector.Normalized()*bound;
    return vector;
  }

  void slew(double target,double &angle,double &rate,double dt) const {
    const double error=target-angle;
    const double cap=std::min(maxRate,std::sqrt(2*maxAngularAcceleration*std::abs(error)));
    const double desired=std::clamp(4*error,-cap,cap);
    rate+=std::clamp(desired-rate,-maxAngularAcceleration*dt,maxAngularAcceleration*dt);
    angle+=rate*dt;
  }

  bool safePosition(const gz::math::Vector3d &p) const {
    return (hubY-p.Y())*axisY+(hubZ-p.Z())*axisZ>=minStandOff &&
        std::abs(p.X())<=150 && p.Z()>=25 && p.Z()<=215;
  }

  void applyExternal(double t,double dt) {
    const double wall=wallSeconds();
    {
      std::lock_guard<std::mutex> lock(commandMutex);
      if(pending.sequence>seenSequence){
        seenSequence=pending.sequence;
        if(t<=pending.validSim && wall<=pending.validWall && safePosition(pending.position) &&
           std::abs(pending.yaw)<=75 && std::abs(pending.pitch)<=65){
          active=pending;appliedAt=t;appliedWall=wall;
        }else if(t>pending.validSim || wall>pending.validWall)expiredSequence=pending.sequence;
      }
    }
    if(active.sequence>=0 && (t>active.validSim || wall>active.validWall)){
      expiredSequence=active.sequence;active.sequence=-1;
      active.position=position;active.yaw=gimbalYaw;active.pitch=gimbalPitch;
    }
    const auto error=active.position-position;
    auto desired=limit(error*2,std::min(maxSpeed,std::sqrt(2*maxAcceleration*error.Length())));
    velocity+=limit(desired-velocity,maxAcceleration*dt);
    const auto next=position+velocity*dt;
    if(!safePosition(next)){safetyFault=true;return;}
    position=next;
    slew(active.yaw,gimbalYaw,yawRate,dt);
    slew(active.pitch,gimbalPitch,pitchRate,dt);
    if(std::abs(gimbalYaw)>75 || std::abs(gimbalPitch)>65)safetyFault=true;
  }

 public:
  void Configure(const Entity &entity,const std::shared_ptr<const sdf::Element> &s,
      gz::sim::EntityComponentManager &ecm,gz::sim::EventManager &) override {
    gz::sim::Model model(entity);
    joint=model.JointByName(ecm,"rotor_joint");
    rotor=model.LinkByName(ecm,"rotor");
    ecm.CreateComponent(joint,gz::sim::components::JointPosition());
    ecm.CreateComponent(joint,gz::sim::components::JointVelocity());
    std::istringstream curve(s->Get<std::string>("omega_points"));
    double time, speed;
    while(curve>>time>>speed){times.push_back(time);speeds.push_back(speed);}
    if (times.empty()) throw std::runtime_error("U3 omega_points is empty");
    baseX=s->Get<double>("base_x");baseY=s->Get<double>("base_y");baseZ=s->Get<double>("base_z");
    yawAmplitude=s->Get<double>("yaw_amplitude_deg");pitch=s->Get<double>("pitch_deg");
    period=s->Get<double>("period_s");travel=s->Get<double>("translation_amplitude_m");
    external=s->HasElement("external_control") && s->Get<bool>("external_control");
    if(external){
      position.Set(baseX,baseY,baseZ);active.position=position;active.pitch=pitch;gimbalPitch=pitch;
      maxSpeed=s->Get<double>("max_speed");maxAcceleration=s->Get<double>("max_acceleration");
      maxRate=s->Get<double>("max_rate");maxAngularAcceleration=s->Get<double>("max_angular_acceleration");
      hubY=s->Get<double>("hub_y");hubZ=s->Get<double>("hub_z");axisY=s->Get<double>("axis_y");axisZ=s->Get<double>("axis_z");
      minStandOff=s->Get<double>("min_standoff");
      node.Subscribe("/u3/command",&U3Motion::command,this);
    }
    publisher=node.Advertise<gz::msgs::StringMsg>("/u3/state");
  }

  void PreUpdate(const gz::sim::UpdateInfo &info,gz::sim::EntityComponentManager &ecm) override {
    if(info.paused)return;
    const double t=std::chrono::duration<double>(info.simTime).count();
    commandOmega=omegaAt(t);
    ecm.SetComponentData<gz::sim::components::JointVelocityCmd>(joint,{commandOmega*rad});
    if(camera==gz::sim::kNullEntity)
      camera=ecm.EntityByComponents(gz::sim::components::Model(),gz::sim::components::Name("camera"));
    if(camera==gz::sim::kNullEntity)return;
    if(external){
      if(!safetyFault)applyExternal(t,std::min(.001,std::max(0.,t-lastTime)));
      lastTime=t;yawRelative=gimbalYaw;
      gz::sim::Model(camera).SetWorldPoseCmd(ecm,gz::math::Pose3d(position,gz::math::Quaterniond(0,gimbalPitch*rad,(90+gimbalYaw)*rad)));
      return;
    }
    const double wave=std::sin(360*rad*t/period);
    yawRelative=yawAmplitude*wave;
    gz::sim::Model(camera).SetWorldPoseCmd(ecm,gz::math::Pose3d(baseX+travel*wave,baseY,baseZ,0,pitch*rad,(90+yawRelative)*rad));
  }

  void PostUpdate(const gz::sim::UpdateInfo &info,const gz::sim::EntityComponentManager &ecm) override {
    if(info.paused || camera==gz::sim::kNullEntity || info.iterations%5!=0)return;
    auto q=ecm.Component<gz::sim::components::JointPosition>(joint);
    auto v=ecm.Component<gz::sim::components::JointVelocity>(joint);
    if(!q || !v || q->Data().empty() || v->Data().empty())return;
    auto cp=gz::sim::worldPose(camera,ecm), rp=gz::sim::worldPose(rotor,ecm);
    std::ostringstream out;
    out<<std::setprecision(17)<<"{\"sim_s\":"<<std::chrono::duration<double>(info.simTime).count()
       <<",\"joint_q_rad\":"<<q->Data()[0]<<",\"joint_v_rad_s\":"<<v->Data()[0]
       <<",\"omega_command_deg_s\":"<<commandOmega<<",\"yaw_command_relative_deg\":"<<yawRelative;
    if(external)out<<",\"executed_sequence\":"<<active.sequence<<",\"command_applied_sim_s\":"<<appliedAt
                   <<",\"command_applied_wall_s\":"<<appliedWall<<",\"expired_sequence\":"<<expiredSequence
                   <<",\"safety_fault\":"<<(safetyFault?"true":"false");
    auto pose=[&out](const char *name,const gz::math::Pose3d &p){
      out<<",\""<<name<<"\":["<<p.Pos().X()<<","<<p.Pos().Y()<<","<<p.Pos().Z()<<","
         <<p.Rot().W()<<","<<p.Rot().X()<<","<<p.Rot().Y()<<","<<p.Rot().Z()<<"]";
    };
    pose("camera_pose",cp);pose("rotor_pose",rp);out<<"}";
    gz::msgs::StringMsg message;message.set_data(out.str());publisher.Publish(message);
  }
};

GZ_ADD_PLUGIN(U3Motion,gz::sim::System,gz::sim::ISystemConfigure,gz::sim::ISystemPreUpdate,gz::sim::ISystemPostUpdate)
