// Atmosphere used by the sky bakes: Rayleigh + hazy aerosol + ozone, single scattering plus a
// cheap multiple-scatter fill. Distances in km. Up is +y, the observer sits at (0,RE+alt,0).
const float RE=6371., RT=6471.;
const float PI=3.14159265;
const vec3 BR=vec3(5.8e-3,13.5e-3,33.1e-3);
const vec3 BO=vec3(.65e-3,1.88e-3,.085e-3);
const float HR=8.,HM=1.15;
uniform vec3 uSunDir;
uniform float uHaze;      // aerosol extinction at sea level, km^-1
uniform float uE0;        // solar constant in scene units
uniform vec3 uWB;         // camera white balance
uniform float uFill;      // multiple scattering fill strength

vec3 bm(){ return uHaze*vec3(.84,1.,1.24); }

vec2 sphere(vec3 ro,vec3 rd,float r){
  float b=dot(ro,rd),c=dot(ro,ro)-r*r,d=b*b-c;
  if(d<0.) return vec2(1e9,-1e9);
  d=sqrt(d); return vec2(-b-d,-b+d);
}

vec3 tauAt(float alt){
  float o=max(1.-abs(alt-25.)/15.,0.);
  return vec3(exp(-alt/HR),exp(-alt/HM),o);
}

// optical depth from p along dir to the top of the atmosphere; large when the ground is in the way
vec3 odSun(vec3 p,vec3 dir){
  vec2 g=sphere(p,dir,RE-.02);
  if(g.x>0.&&g.x<g.y) return vec3(100.);
  float tm=sphere(p,dir,RT).y;
  const int N=14;
  vec3 s=vec3(0.);
  float prev=0.;
  for(int i=0;i<N;i++){
    float a=float(i+1)/float(N); float t=tm*a*a;
    float mid=.5*(t+prev);
    float alt=length(p+dir*mid)-RE;
    s+=tauAt(max(alt,0.))*(t-prev);
    prev=t;
  }
  return s;
}

vec3 extinction(vec3 od){ return BR*od.x+bm()*od.y+BO*od.z; }

float phaseR(float c){ return 3./(16.*PI)*(1.+c*c); }
float hg(float c,float g){ float g2=g*g; return (1.-g2)/(4.*PI*pow(max(1.+g2-2.*g*c,1e-4),1.5)); }
float phaseM(float c){ return .78*hg(c,.76)+.1*hg(c,.93)+.12*hg(c,-.25); }

// in-scattered radiance on [0,tmax] along a ray; T is the transmittance of that segment
vec3 scatter(vec3 ro,vec3 rd,float tmax,out vec3 T,int N){
  float c=dot(rd,uSunDir);
  float pr=phaseR(c),pm=phaseM(c);
  vec3 od=vec3(0.),L=vec3(0.);
  float prev=0.;
  // sky glow that lights the air from all directions at low sun
  vec3 fillC=uFill*vec3(.42,.56,.86);
  for(int i=0;i<N;i++){
    float a=float(i+1)/float(N); float t=tmax*a*a;
    float dt=t-prev; float mid=.5*(t+prev);
    vec3 p=ro+rd*mid;
    float alt=max(length(p)-RE,0.);
    vec3 d=tauAt(alt);
    vec3 odMid=od+.5*d*dt;
    vec3 ts=exp(-extinction(odSun(p,uSunDir)));
    vec3 tv=exp(-extinction(odMid));
    vec3 sR=BR*d.x, sM=bm()*d.y;
    L+=tv*(ts*(sR*pr+sM*pm)+fillC*(sR+.5*sM)*exp(-alt/30.))*dt;
    od+=d*dt;
    prev=t;
  }
  T=exp(-extinction(od));
  return L*uE0*uWB;
}

vec3 panoDir(vec2 uv){
  float az=(uv.x-.5)*2.*PI;
  float el=.5*PI*uv.y*uv.y;
  float ce=cos(el);
  return vec3(sin(az)*ce,sin(el),cos(az)*ce);
}
