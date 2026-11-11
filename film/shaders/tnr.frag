#version 410
// Motion-adaptive temporal denoise: the phone smears a little in the dark.
#include "common.glsl"
in vec2 uv; out vec4 o;
uniform sampler2D uCur, uPrev;
uniform float uStrength, uSigma, uReset;
void main(){
  vec3 c=texture(uCur,uv).rgb;
  if(uReset>.5){ o=vec4(c,1.); return; }
  vec3 cb=textureLod(uCur,uv,2.).rgb;
  vec3 pb=textureLod(uPrev,uv,2.).rgb;
  vec3 p=texture(uPrev,uv).rgb;
  float d=length(cb-pb)/(uSigma+.25*luma(cb)+1e-4);
  float a=uStrength*exp(-d*d);
  o=vec4(mix(c,p,a),1.);
}
