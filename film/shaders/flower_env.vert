#version 410
// the flower again, from where the drop is
in vec3 in_p,in_n,in_t,in_b; in vec2 in_uv; in float in_w;
uniform mat3 uSway; uniform vec3 uPivot; uniform float uFlut;
uniform vec3 uEye,uFaceF,uFaceR,uFaceU;
uniform float uTanE;
out vec3 vP,vN,vT,vB; out vec2 vUv; out float vZ,vL;
const float NEAR=.0012,FAR=4.;
void main(){
  vec3 p=in_p+in_n*in_w*uFlut;
  p=uPivot+uSway*(p-uPivot);
  vP=p; vN=uSway*in_n; vT=uSway*in_t; vB=uSway*in_b; vUv=in_uv;
  vZ=1.; vL=0.;
  vec3 q=p-uEye;
  float z=dot(q,uFaceF);
  float A=(FAR+NEAR)/(FAR-NEAR),B=-2.*FAR*NEAR/(FAR-NEAR);
  gl_Position=vec4(dot(q,uFaceR)/uTanE,dot(q,uFaceU)/uTanE,A*z+B,z);
}
