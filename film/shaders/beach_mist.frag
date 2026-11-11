#version 410
// Salt in the air over the surf, for the views facing the sun: real matter along the ray, not a veil over the picture.
// Where the water is breaking it goes up, leaning downwind and spreading as it rises, torn by a moving 3D field; lower
// and everywhere a thin haze thickens towards the water. All of it is drops, so all of it lights up against the sun.
// The camera only ever turns, so a ray's samples can stay on fixed cylinders round it: thin sheets of air, each one
// sharp, with nothing to show the steps between them. rgb is the light added along the ray, a what it lets through.
in vec2 uv; out vec4 o;
#include "common.glsl"
const float PI=3.14159265;
uniform float uAspect,uTan,uT,uH,uFw,uAmount;
uniform vec3 uF,uR,uU,uSun,uSunE;
uniform vec3 uDrift[3];
uniform vec4 uSkyWin,uPol;
uniform ivec2 uNS;
uniform sampler2D uSrc,uEnv;
uniform sampler2DArray uCloud;
uniform sampler3D uFract;
uniform vec4 uStack[4];          // azimuth, range, half width, seed
const vec2 WIND=vec2(-1.3,3.1);  // m/s, off the land and across
const float RISE=1.5;            // m/s a plume climbs
const float R_NEAR=16.,R_FAR=2600.;
const int N=40;

float hg(float c,float g){ float g2=g*g; return (1.-g2)/(4.*PI*pow(max(1.+g2-2.*g*c,1e-4),1.5)); }

// how much of the sun the clouds let through right now
float sunThrough(){
  float T=1.;
  for(int k=0;k<3;k++){
    vec3 dk=normalize(uSun+uDrift[k]*max(uSun.y,0.));
    vec2 w=vec2((atan(dk.x,dk.z)-uSkyWin.x)/(uSkyWin.y-uSkyWin.x),(asin(dk.y)-uSkyWin.z)/(uSkyWin.w-uSkyWin.z));
    T*=textureLod(uCloud,vec3(w,float(k)),0.).a;
  }
  return T;
}

// what is in the air at this point
float air(vec3 p,float r){
  float y=p.y;
  float plume=0.;
  if(r<580.&&y<70.){
    // what is this high left the water y/RISE ago, and the lip that threw it has come on towards the beach since:
    // what the plume stands on is where that lip is now, so it hangs back behind the wave and does not ride it
    vec2 qs=p.xz-(WIND+vec2(.03,1.)*3.9)*y/RISE;
    float rs=length(qs),ts=atan(qs.x,qs.y);
    float lod=log2(max((.9+.3*y)/(rs*uPol.y*2.),1.));
    vec2 uvS=vec2((ts-uPol.x)/(uPol.y*float(uNS.x)),(log(rs)-uPol.z)/(uPol.w*float(uNS.y)));
    if(uvS.x>0.&&uvS.x<1.&&uvS.y>0.&&uvS.y<1.){
      float b=textureLod(uSrc,uvS,lod).r;
      plume=b*sqrt(b)*2.*exp(-y/(1.4+6.*b));
    }
  }
  vec3 P=vec3(p.x,y*1.5,p.z);
  vec3 flow=vec3(WIND.x,RISE*1.5,WIND.y)*uT;
  float lodN=log2(max(r*uFw*2./.42,1.));
  float n1=textureLod(uFract,(P-flow)/108.,lodN).r;
  float n2=textureLod(uFract,(P-flow*1.35+vec3(31.,0.,17.))/43.,lodN+.6).r;
  float wisp=mix(.25,1.,smoothstep(.34,.72,n1*.72+n2*.28));
  // haze: off the surf and over the water, thickest in the last few metres
  float haze=(.00012*exp(-y/7.)+.000028*exp(-y/30.))*(.55+.9*n1)*smoothstep(30.,160.,r);
  return plume*wisp*.036+haze;
}

// The swell against the rock. Out past the reach of the sim it keeps its own time: a wave every ten seconds or so finds each
// stretch of a foot at a moment of its own, most of them a slop of white and every so often one that goes up. Each goes up
// fast, hangs, leans over and blows away. Far enough off for one flat sheet at the range of the rock.
float footWater(float th,float tanEl){
  float dens=0.;
  for(int i=0;i<4;i++){
    vec4 s=uStack[i];
    float x=(th-s.x)*s.y,reach=s.z*1.75+26.;
    float y=uH+s.y*tanEl;
    if(abs(x)>reach||y<0.||y>64.) continue;
    const float W=15.;
    float id0=floor(x/W),b=0.,wash=0.;
    for(int k=-2;k<=2;k++){
      float id=id0+float(k);
      float h=hash21(vec2(id,s.w));
      float P=10.*(.9+.2*h);
      float T=uT+P*fract(h*7.31)+7.;
      float n=floor(T/P),a=T-n*P;
      float set=hash21(vec2(id*1.7+n*3.1,s.w+2.));
      float H=set>.84?mix(13.,26.,(set-.84)/.16):mix(1.2,5.5,set*set/.7);
      float top=H*(1.-exp(-a/.8));
      float cx=(id+.5+.8*(fract(h*19.7)-.5))*W+WIND.x*a*min(y/(H+1.),1.)*.8;
      float wid=W*.24+.55*a+.2*y;
      float xx=(x-cx)/wid;
      b+=exp(-xx*xx)*smoothstep(top*1.02,top*.3,y)*exp(-max(a-.5,0.)/mix(1.5,3.2,smoothstep(.84,1.,set)))/(.75+.11*top);
      wash=max(wash,exp(-xx*xx*.3)*exp(-a/2.6));
    }
    float lod=log2(max(s.y*uFw*2./.42,1.));
    float f=textureLod(uFract,vec3(x-WIND.x*uT,(y-1.4*uT)*1.15,s.w*9.)/22.,lod).r;
    // the white the waves leave climbing the foot, a metre or two of it, always somewhere and never the same place
    float ln=vnoise(vec2(x/9.+s.w,uT*.13))*.55+vnoise(vec2(x/3.,uT*.31+s.w))*.45+.3*wash;
    float line=smoothstep(.22+1.5*wash,.03,y)*smoothstep(.5,.8,ln);
    // it is confined to the rock: past the ends of the foot there is nothing for the swell to hit
    float foot=1.-smoothstep(s.z*1.15,s.z*1.75,abs(x));
    dens+=foot*(6.*b*smoothstep(.2,.62,f+.55*min(b,1.)-.28)+.75*line);
  }
  return dens;
}

void main(){
  vec2 v=(uv*2.-1.)*vec2(uAspect,1.);
  vec3 d=normalize(uF+v.x*uTan*uR+v.y*uTan*uU);
  float tanEl=d.y/sqrt(max(1.-d.y*d.y,1e-6));
  float th=atan(d.x,d.z);
  vec2 nr=vec2(sin(th),cos(th));
  // on to the water, or past the stacks
  float rEnd=tanEl<-.0017?uH/-tanEl:R_FAR;
  float rho0=log(R_NEAR),dr=(log(R_FAR)-rho0)/float(N),rhoE=log(rEnd);
  float cc=dot(d,uSun);
  float ph=.6*hg(cc,.75)+.34*hg(cc,.4)+.06/(4.*PI);
  vec3 sun=uSunE*sunThrough()*ph;
  vec3 amb=textureLod(uEnv,vec2(.5,.35),6.).rgb*.75+textureLod(uEnv,vec2(.62,.12),5.).rgb*.25;
  vec3 J=sun+amb*vec3(.86,.93,1.1);
  float tau=0.;
  vec3 S=vec3(0.);
  for(int i=0;i<N;i++){
    float rho=rho0+(float(i)+.5)*dr;
    float w=clamp((rhoE-(rho-.5*dr))/dr,0.,1.);
    if(w<=0.) break;
    float r=exp(rho);
    float y=uH+r*tanEl;
    if(y<=0.) y=.01;
    float dt=air(vec3(r*nr.x,y,r*nr.y),r)*dr*r*w;
    S+=J*exp(-tau)*(1.-exp(-dt));
    tau+=dt;
  }
  if(tanEl<.05){
    float fb=footWater(th,tanEl);
    if(fb>0.){ S+=J*exp(-tau)*(1.-exp(-fb)); tau+=fb; }
  }
  o=vec4(S*uAmount,exp(-tau*uAmount));
}
