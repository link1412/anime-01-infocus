#version 410
// light, per texel of the sheet. what a lens does to it comes later; this is what is there.
in vec3 vP,vN,vT,vB; in vec2 vUv;
out vec4 o;
const int M=32;
uniform sampler2D uPig,uDet,uZ1,uZ2;
uniform vec3 uLd[M];        // direction of travel of the light
uniform vec3 uLe[M];        // irradiance on a surface facing it
uniform vec3 uLx[M],uLy[M];
uniform vec3 uLo;
uniform vec2 uLs;           // half extent across, half extent along
uniform vec4 uTile[M];      // origin xy, size xy, in atlas uv
uniform vec3 uTocc;         // what a sheet in the way lets through
uniform vec3 uC,uFill;
uniform float uTrans,uRefl,uKDet,uForward;

void main(){
  vec4 pig=texture(uPig,vUv);
  vec4 det=texture(uDet,vUv);
  vec3 Ng=normalize(vN);
  vec3 N=normalize(Ng-uKDet*(det.x*normalize(vT)+det.y*normalize(vB)));
  vec3 V=normalize(uC-vP);
  if(dot(Ng,V)<0.){ N=-N; Ng=-Ng; }
  // the colour is a transmittance; thicker means deeper
  vec3 T1=pow(max(pig.rgb,vec3(.001)),vec3(det.a));
  vec3 Tr=T1*uTrans;
  vec3 Rf=(T1*.92+.06*vec3(.9,.8,.7))*uRefl;
  vec3 L=vec3(0.);
  for(int i=0;i<M;i++){
    vec3 d=uLd[i];
    vec3 q=vP-uLo;
    vec2 xy=vec2(dot(q,uLx[i]),dot(q,uLy[i]))/uLs.x*.5+.5;
    float z=dot(q,d);
    vec2 a=uTile[i].xy+clamp(xy,0.,1.)*uTile[i].zw;
    float z1=texture(uZ1,a).r,z2=texture(uZ2,a).r;
    const float BIAS=.0011;
    vec3 vis=vec3(1.);
    if(z>z1+BIAS){ vis*=uTocc; if(z>z2+BIAS) vis*=uTocc; }
    float nl=-dot(N,d);      // >0: the light arrives on the side we are looking at
    // a petal is a few cells thick and they pass the light sideways: no fold goes truly dark
    float wr=(abs(nl)+.30)/1.30;
    // through the sheet: scattered, and a little more in the direction it was going
    float fw=pow(max(dot(d,-V),0.),3.);
    vec3 f=mix(Tr*(1.+uForward*fw),Rf,smoothstep(-.25,.25,nl))*wr;
    L+=uLe[i]*vis*f;
  }
  // what neighbouring sheets give each other, and what is left in a fold
  float cav=det.z;
  L+=uFill*T1*(1.-.3*cav);
  o=vec4(L/3.14159,1.);
}
