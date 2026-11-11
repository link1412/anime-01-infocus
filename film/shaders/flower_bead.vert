#version 410
#include "flower_common.glsl"
in vec2 p;
in vec3 ic,inb,it,ib; in vec4 ishape; in vec4 iuv; in vec2 iw;
uniform mat3 uSway; uniform vec3 uPivot; uniform float uFlut;
out vec3 vC,vNb,vTb,vBb; out vec4 vS; out vec4 vU; out vec2 vW;
void main(){
  vec3 c=ic+inb*iw.x*uFlut;
  c=uPivot+uSway*(c-uPivot);
  vC=c; vNb=uSway*inb; vTb=uSway*it; vBb=uSway*ib; vS=ishape; vU=iuv; vW=iw;
  float R=ishape.x*max(ishape.y,1.)*1.25;
  vec4 q=project(c+vNb*ishape.x*.5);
  float z=q.w;
  float s=R/(z*uTan)*uZoom;
  vec2 off=p*s/vec2(uAspect,1.);
  gl_Position=vec4(q.xy+off*z,q.z,q.w);
}
