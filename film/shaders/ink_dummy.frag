#version 410
// A stand-in for the shot that comes before this world: a golden sea, in linear light.
#include "common.glsl"
in vec2 uv; out vec4 o;
uniform float uAspect,uT,uHorizon,uSunR;
uniform vec2 uSunC;
uniform sampler2D uNoise;
void main(){
  vec2 p=vec2((uv.x*2.-1.)*uAspect,uv.y*2.-1.);
  float dS=length(p-uSunC);
  vec3 c;
  if(p.y>uHorizon){
    float a=(p.y-uHorizon)/(1.-uHorizon);
    c=mix(vec3(2.3,1.05,.34),vec3(.62,.46,.42),pow(a,.62));
    c+=vec3(2.6,1.15,.30)*exp(-dS*dS*3.2)+vec3(1.2,.5,.12)*exp(-dS*1.4);
  }else{
    float z=clamp((uHorizon-p.y)/(uHorizon+1.),0.,1.);
    float persp=1./(z+.03);
    vec2 q=vec2(p.x*persp*.09+uT*.002,persp*.09-uT*.01);
    float w=texture(uNoise,q).g*.6+texture(uNoise,q*2.3+.2).g*.4;
    c=mix(vec3(.36,.18,.09),vec3(.085,.05,.05),pow(z,.7));
    float path=exp(-pow((p.x-uSunC.x)/(.14+z*.46),2.));
    c+=vec3(3.0,1.5,.5)*path*smoothstep(.56,.84,w)*(1.2-z);
    c+=vec3(.5,.26,.08)*(1.-z)*.35;
    c*=.72+.56*w;
  }
  c=mix(c,vec3(30.,20.,9.),1.-smoothstep(uSunR*.97,uSunR*1.03,dS));
  o=vec4(c,1.);
}
