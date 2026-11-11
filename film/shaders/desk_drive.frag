#version 410
// How much light the screen is giving, per corner of the rectangle: the weights of the baked passes for this frame.
#include "common.glsl"
in vec2 uv; out vec4 o;
uniform sampler2D uScreen;
uniform float uIsFeed,uBright,uLeak,uGain;
void main(){
  vec2 corner=step(.5,uv);
  float lod=max(log2(float(textureSize(uScreen,0).x)/24.),0.);
  vec3 s=vec3(0.); float w=0.;
  for(int j=0;j<8;j++) for(int i=0;i<12;i++){
    vec2 st=(vec2(i,j)+.5)/vec2(12.,8.);
    float k=(corner.x>.5?st.x:1.-st.x)*(corner.y>.5?st.y:1.-st.y);
    vec2 q=st;
    vec3 e=vec3(0.);
    if(uIsFeed>.5){
      q.x=(st.x-.5)*(16./9.)/(4./3.)+.5;
      if(q.x>0.&&q.x<1.) e=srgb2lin(textureLod(uScreen,q,lod).rgb);
    } else e=srgb2lin(textureLod(uScreen,q,lod).rgb);
    s+=e*k; w+=k;
  }
  // a dark terminal still lights the room: the backlight never quite shuts, and the camera is wide open
  o=vec4((s/w*uGain*uBright+uLeak*vec3(.92,.97,1.08)*step(.001,uBright)),1.);
}
